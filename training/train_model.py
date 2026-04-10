"""
FCL Freight Rate Forecasting Model — Model Training
=====================================================
Trains two complementary models on the WCI composite index:

  1. SARIMAX  — captures seasonality, trend, and exogenous macro drivers
  2. XGBoost  — captures nonlinear interactions and feature importance

The WCI composite (USD/40ft container, monthly) is used as the target variable
as it is the best freely available proxy for FCL spot market rates.

Outputs:
  /models/sarimax_model.pkl
  /models/xgb_model.pkl
  /models/model_metadata.json
  /outputs/training_summary.csv
"""

import os
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
TARGET_COL = 'wci_composite'

# SARIMAX exogenous features (confirmed available for full training period)
SARIMAX_EXOG = [
    'brent_crude_usd',
    'usdcny',
    'us_indpro',
    'us_cfnai',
    'bdry_etf',
    'china_exports_usd',
]

# XGBoost features (includes lags and derived features)
XGB_FEATURES = [
    'brent_crude_usd', 'brent_crude_usd_lag1', 'brent_crude_usd_lag2',
    'usdcny', 'usdcny_lag1',
    'us_indpro', 'us_cfnai', 'us_mfg_orders',
    'bdry_etf', 'bdry_etf_lag1', 'bdry_etf_lag2',
    'china_exports_usd', 'us_imports_china',
    'wci_composite_lag1', 'wci_composite_lag2',
    'brent_crude_usd_ma3', 'bdry_etf_ma3', 'wci_composite_ma3',
    'month',
]

# Training split: use 2021-01 to 2023-12 for training, 2024 for test
TRAIN_END = '2023-12'
TEST_START = '2024-01'


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and preparation
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, 'features_monthly.parquet')
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    log.info(f"Loaded data: {df.shape}, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def stationarity_test(series: pd.Series, name: str) -> dict:
    """Run ADF and KPSS tests for stationarity."""
    result = {}
    try:
        adf = adfuller(series.dropna(), autolag='AIC')
        result['adf_stat'] = round(adf[0], 4)
        result['adf_pvalue'] = round(adf[1], 4)
        result['adf_stationary'] = adf[1] < 0.05
    except Exception as e:
        result['adf_error'] = str(e)
    try:
        kpss_stat, kpss_p, _, _ = kpss(series.dropna(), regression='c', nlags='auto')
        result['kpss_stat'] = round(kpss_stat, 4)
        result['kpss_pvalue'] = round(kpss_p, 4)
        result['kpss_stationary'] = kpss_p > 0.05
    except Exception as e:
        result['kpss_error'] = str(e)
    log.info(f"  {name}: ADF p={result.get('adf_pvalue','?')} | KPSS p={result.get('kpss_pvalue','?')}")
    return result


def prepare_sarimax_data(df: pd.DataFrame):
    """Prepare data for SARIMAX: fill missing exog with forward-fill."""
    # Use only columns that are available
    avail_exog = [c for c in SARIMAX_EXOG if c in df.columns]
    
    data = df[[TARGET_COL] + avail_exog].copy()
    # Forward-fill then backward-fill any gaps
    data = data.ffill().bfill()
    
    # Log-transform target to stabilise variance
    data['target_log'] = np.log(data[TARGET_COL])
    
    train = data.loc[:TRAIN_END]
    test  = data.loc[TEST_START:]
    
    return train, test, avail_exog


def prepare_xgb_data(df: pd.DataFrame):
    """Prepare data for XGBoost: select available features, fill NaN."""
    avail_feats = [c for c in XGB_FEATURES if c in df.columns]
    
    data = df[[TARGET_COL] + avail_feats].copy()
    data = data.ffill().bfill()
    data = data.dropna()
    
    train = data.loc[:TRAIN_END]
    test  = data.loc[TEST_START:]
    
    return train, test, avail_feats


# ─────────────────────────────────────────────────────────────────────────────
# SARIMAX training
# ─────────────────────────────────────────────────────────────────────────────
def train_sarimax(train: pd.DataFrame, exog_cols: list) -> dict:
    """
    Fit SARIMAX(1,1,1)(1,0,1,12) on log-transformed WCI.
    Order selection based on AIC grid search over p,q ∈ {0,1,2}.
    """
    log.info("\n--- Training SARIMAX ---")
    
    y_train = train['target_log']
    X_train = train[exog_cols]
    
    # Standardise exogenous variables
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    
    best_aic = np.inf
    best_model = None
    best_order = None
    best_seasonal = None
    
    # Grid search over ARIMA orders
    for p in [0, 1, 2]:
        for q in [0, 1, 2]:
            for d in [0, 1]:
                for P in [0, 1]:
                    for Q in [0, 1]:
                        try:
                            mod = SARIMAX(
                                y_train,
                                exog=X_train_scaled,
                                order=(p, d, q),
                                seasonal_order=(P, 0, Q, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            )
                            res = mod.fit(disp=False, maxiter=200)
                            if res.aic < best_aic:
                                best_aic = res.aic
                                best_model = res
                                best_order = (p, d, q)
                                best_seasonal = (P, 0, Q, 12)
                        except Exception:
                            continue
    
    log.info(f"  Best SARIMAX order: {best_order} x {best_seasonal}, AIC={best_aic:.2f}")
    
    return {
        'model': best_model,
        'scaler': scaler,
        'exog_cols': exog_cols,
        'order': best_order,
        'seasonal_order': best_seasonal,
        'aic': best_aic,
    }


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost training
# ─────────────────────────────────────────────────────────────────────────────
def train_xgboost(train: pd.DataFrame, feat_cols: list) -> dict:
    """Train XGBoost regressor on WCI (level, not log)."""
    log.info("\n--- Training XGBoost ---")
    
    X_train = train[feat_cols]
    y_train = train[TARGET_COL]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train_scaled, y_train)
    
    # Feature importance
    importance = pd.Series(xgb.feature_importances_, index=feat_cols)
    importance = importance.sort_values(ascending=False)
    log.info(f"  Top 5 features: {importance.head(5).to_dict()}")
    
    return {
        'model': xgb,
        'scaler': scaler,
        'feat_cols': feat_cols,
        'feature_importance': importance.to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(y_true: pd.Series, y_pred: pd.Series, name: str) -> dict:
    """Compute MAE, RMSE, MAPE, and directional accuracy."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional accuracy
    actual_dir = np.sign(y_true.diff().dropna())
    pred_dir   = np.sign(pd.Series(y_pred, index=y_true.index).diff().dropna())
    dir_acc    = (actual_dir == pred_dir).mean() * 100
    
    log.info(f"\n  {name} Test Performance (2024):")
    log.info(f"    MAE:  {mae:.1f} USD/40ft")
    log.info(f"    RMSE: {rmse:.1f} USD/40ft")
    log.info(f"    MAPE: {mape:.1f}%")
    log.info(f"    Dir Accuracy: {dir_acc:.1f}%")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'dir_acc': dir_acc}


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("FCL Forecast — Model Training")
    log.info("=" * 60)
    
    # Load data
    df = load_data()
    
    # Stationarity tests on target
    log.info("\nStationarity tests:")
    stat_results = {}
    stat_results[TARGET_COL] = stationarity_test(df[TARGET_COL], TARGET_COL)
    stat_results[f'{TARGET_COL}_diff'] = stationarity_test(
        df[TARGET_COL].diff().dropna(), f'{TARGET_COL}_diff'
    )
    
    # ── SARIMAX ──────────────────────────────────────────────────────────────
    train_s, test_s, exog_cols = prepare_sarimax_data(df)
    sarimax_result = train_sarimax(train_s, exog_cols)
    
    # SARIMAX in-sample fit
    fitted_log = sarimax_result['model'].fittedvalues
    fitted_level = np.exp(fitted_log)
    train_actual = np.exp(train_s['target_log'])
    
    # SARIMAX out-of-sample forecast for 2024
    X_test_s = test_s[exog_cols]
    X_test_scaled = pd.DataFrame(
        sarimax_result['scaler'].transform(X_test_s),
        index=X_test_s.index,
        columns=X_test_s.columns
    )
    forecast_log = sarimax_result['model'].forecast(
        steps=len(test_s),
        exog=X_test_scaled
    )
    sarimax_pred = np.exp(forecast_log)
    sarimax_pred.index = test_s.index
    sarimax_metrics = evaluate_model(test_s[TARGET_COL], sarimax_pred, 'SARIMAX')
    
    # ── XGBoost ──────────────────────────────────────────────────────────────
    train_x, test_x, feat_cols = prepare_xgb_data(df)
    xgb_result = train_xgboost(train_x, feat_cols)
    
    X_test_x = test_x[feat_cols]
    X_test_x_scaled = xgb_result['scaler'].transform(X_test_x)
    xgb_pred = pd.Series(
        xgb_result['model'].predict(X_test_x_scaled),
        index=test_x.index,
        name='xgb_pred'
    )
    xgb_metrics = evaluate_model(test_x[TARGET_COL], xgb_pred, 'XGBoost')
    
    # ── Ensemble (equal weight) ───────────────────────────────────────────────
    # Align indices
    common_idx = sarimax_pred.index.intersection(xgb_pred.index)
    ensemble_pred = (sarimax_pred.loc[common_idx] + xgb_pred.loc[common_idx]) / 2
    ensemble_metrics = evaluate_model(
        test_s.loc[common_idx, TARGET_COL], ensemble_pred, 'Ensemble (50/50)'
    )
    
    # ── Save models ───────────────────────────────────────────────────────────
    sarimax_path = os.path.join(MODEL_DIR, 'sarimax_model.pkl')
    xgb_path     = os.path.join(MODEL_DIR, 'xgb_model.pkl')
    
    with open(sarimax_path, 'wb') as f:
        pickle.dump(sarimax_result, f)
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_result, f)
    
    log.info(f"\nSaved: {sarimax_path}")
    log.info(f"Saved: {xgb_path}")
    
    # ── Save metadata ─────────────────────────────────────────────────────────
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'target': TARGET_COL,
        'train_period': f"2021-01 to {TRAIN_END}",
        'test_period': f"{TEST_START} to 2024-12",
        'sarimax': {
            'order': list(sarimax_result['order']),
            'seasonal_order': list(sarimax_result['seasonal_order']),
            'aic': round(sarimax_result['aic'], 2),
            'exog_cols': exog_cols,
            **{k: round(v, 2) for k, v in sarimax_metrics.items()},
        },
        'xgboost': {
            'n_estimators': 300,
            'max_depth': 4,
            'feat_cols': feat_cols,
            'top_features': dict(list(xgb_result['feature_importance'].items())[:10]),
            **{k: round(v, 2) for k, v in xgb_metrics.items()},
        },
        'ensemble': {
            **{k: round(v, 2) for k, v in ensemble_metrics.items()},
        },
        'stationarity': stat_results,
    }
    
    meta_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f"Saved: {meta_path}")
    
    # ── Save predictions ──────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        'actual': test_s[TARGET_COL],
        'sarimax_pred': sarimax_pred,
        'xgb_pred': xgb_pred,
        'ensemble_pred': ensemble_pred,
    })
    pred_path = os.path.join(OUTPUT_DIR, 'test_predictions.csv')
    pred_df.to_csv(pred_path)
    log.info(f"Saved: {pred_path}")
    
    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)
    log.info(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Dir%':>8}")
    log.info("-" * 60)
    for name, m in [('SARIMAX', sarimax_metrics), ('XGBoost', xgb_metrics), ('Ensemble', ensemble_metrics)]:
        log.info(f"{name:<20} {m['mae']:>8.0f} {m['rmse']:>8.0f} {m['mape']:>7.1f}% {m['dir_acc']:>7.1f}%")
    
    return metadata


if __name__ == '__main__':
    metadata = main()
    print("\nModel metadata:")
    print(json.dumps(metadata, indent=2, default=str))
