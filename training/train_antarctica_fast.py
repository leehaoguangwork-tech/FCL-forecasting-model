"""
Per-Lane SARIMAX + XGBoost Hybrid — 15 Representative Lanes (Fast Demo)
=======================================================================
15 lanes covering: Asia, Europe, Americas, Middle East, Africa, Oceania
Once validated, scale to all 110 lanes.
"""

import os, json, warnings, pickle, sys, time
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
MDL  = os.path.join(BASE, 'models', 'antarctica')
OUT  = os.path.join(BASE, 'outputs', 'antarctica')
os.makedirs(MDL, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ── 15 representative lanes (geography-diverse) ───────────────────────────────
DEMO_LANES = [
    'CNSHA',   # China – Shanghai (Asia, high volume)
    'JPTYO',   # Japan – Tokyo (Asia)
    'KRPUS',   # South Korea – Busan (Asia)
    'INNSA',   # India – Nhava Sheva (South Asia)
    'THBKK',   # Thailand – Bangkok (SE Asia)
    'DEHAM',   # Germany – Hamburg (Europe)
    'NLRTM',   # Netherlands – Rotterdam (Europe)
    'GBFXT',   # UK – Felixstowe (Europe)
    'USNYC',   # USA – New York (Americas)
    'USLAX',   # USA – Los Angeles (Americas)
    'ARBUE',   # Argentina – Buenos Aires (S. America)
    'AEJEA',   # UAE – Jebel Ali (Middle East)
    'NGLOS',   # Nigeria – Lagos (Africa)
    'AUSYD',   # Australia – Sydney (Oceania)
    'QAHMD',   # Qatar – Hamad (Middle East)
]

panel = pd.read_parquet(os.path.join(DATA, 'antarctica_monthly_panel.parquet'))
exog  = pd.read_parquet(os.path.join(DATA, 'exog_features_antarctica.parquet'))

# Only keep lanes that exist in the panel
DEMO_LANES = [l for l in DEMO_LANES if l in panel.columns]
print(f"Demo lanes: {len(DEMO_LANES)} | {DEMO_LANES}", flush=True)

exog = exog.reindex(panel.index).ffill().bfill()

EXOG_COLS = ['brent_crude', 'usdcny', 'us_indpro', 'us_cfnai', 'bdry_etf',
             'dummy_covid', 'dummy_supply_crunch', 'dummy_ukraine',
             'dummy_red_sea', 'dummy_hormuz']

SARIMAX_ORDER  = (1, 1, 1)
SARIMAX_SORDER = (0, 1, 1, 12)
MIN_TRAIN = 24

def make_xgb_features(y_series, exog_df):
    df = pd.DataFrame(index=y_series.index)
    for lag in [1, 2, 3, 6, 12]:
        df[f'rate_lag{lag}'] = y_series.shift(lag)
    df['rate_roll3'] = y_series.shift(1).rolling(3).mean()
    df['rate_roll6'] = y_series.shift(1).rolling(6).mean()
    df['month_sin']  = np.sin(2 * np.pi * y_series.index.month / 12)
    df['month_cos']  = np.cos(2 * np.pi * y_series.index.month / 12)
    for col in EXOG_COLS:
        if col in exog_df.columns:
            df[col]         = exog_df[col].values
            df[f'{col}_l1'] = exog_df[col].shift(1).values
    return df

def safe_metrics(act, pred):
    act  = np.array(act,  dtype=float)
    pred = np.array(pred, dtype=float)
    mask = np.isfinite(act) & np.isfinite(pred)
    act  = act[mask]; pred = pred[mask]
    if len(act) < 2:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'dir_acc': np.nan}
    pred = np.clip(pred, 0, act.max() * 10)
    mae  = float(mean_absolute_error(act, pred))
    rmse = float(np.sqrt(mean_squared_error(act, pred)))
    mape = float(np.mean(np.abs((act - pred) / np.maximum(act, 1))) * 100)
    da   = float(np.mean(np.sign(np.diff(act)) == np.sign(np.diff(pred))) * 100)
    return {'mae': round(mae,2), 'rmse': round(rmse,2),
            'mape': round(mape,2), 'dir_acc': round(da,2)}

# ── Walk-forward backtest ─────────────────────────────────────────────────────
results = {}
all_bt  = []
print(f"\nRunning walk-forward backtest for {len(DEMO_LANES)} lanes...", flush=True)
t0 = time.time()

for i, lane in enumerate(DEMO_LANES):
    tl = time.time()
    y = panel[lane].dropna()
    if len(y) < MIN_TRAIN + 6:
        print(f"  [{i+1:2d}/{len(DEMO_LANES)}] {lane}: SKIP (only {len(y)} months)", flush=True)
        continue

    exog_lane = exog.reindex(y.index)[EXOG_COLS].ffill().bfill()
    xgb_feat  = make_xgb_features(y, exog_lane)
    y_log     = np.log1p(y)

    actuals, sar_preds, stk_preds, dates = [], [], [], []
    xgb_model = None
    xgb_last_trained = -999

    # Fit SARIMAX once on first MIN_TRAIN months
    try:
        m0 = SARIMAX(y_log.iloc[:MIN_TRAIN],
                     exog=exog_lane.iloc[:MIN_TRAIN],
                     order=SARIMAX_ORDER,
                     seasonal_order=SARIMAX_SORDER,
                     enforce_stationarity=False,
                     enforce_invertibility=False)
        r0 = m0.fit(disp=False, maxiter=100, method='lbfgs')
    except Exception as e:
        print(f"  [{i+1:2d}] {lane}: initial SARIMAX fit FAILED: {e}", flush=True)
        continue

    for t in range(MIN_TRAIN, len(y)):
        try:
            r_ext = r0.apply(y_log.iloc[:t], exog=exog_lane.iloc[:t], refit=False)
            ex_te    = exog_lane.iloc[[t]]
            fc_log   = float(r_ext.forecast(steps=1, exog=ex_te).iloc[0])
            sar_pred = float(np.expm1(fc_log))
            insample = np.expm1(r_ext.fittedvalues.values)
            resid    = y.iloc[:t].values - insample
        except Exception:
            continue

        # Retrain XGBoost every 6 steps
        if t - xgb_last_trained >= 6 and t >= MIN_TRAIN + 6:
            try:
                feat_tr = xgb_feat.iloc[:t].copy()
                feat_tr['sar_pred'] = insample
                feat_tr = feat_tr.dropna()
                resid_al = resid[-len(feat_tr):]
                if len(feat_tr) >= 12:
                    xgb_model = XGBRegressor(n_estimators=80, max_depth=3,
                                             learning_rate=0.1, subsample=0.8,
                                             colsample_bytree=0.8,
                                             random_state=42, verbosity=0)
                    xgb_model.fit(feat_tr, resid_al)
                    xgb_last_trained = t
            except Exception:
                xgb_model = None

        if xgb_model is not None:
            try:
                feat_te = xgb_feat.iloc[[t]].copy()
                feat_te['sar_pred'] = sar_pred
                feat_te = feat_te.fillna(0)
                corr = float(xgb_model.predict(feat_te)[0])
                stk_pred = sar_pred + corr
            except Exception:
                stk_pred = sar_pred
        else:
            stk_pred = sar_pred

        if np.isfinite(sar_pred) and np.isfinite(stk_pred):
            actuals.append(float(y.iloc[t]))
            sar_preds.append(sar_pred)
            stk_preds.append(stk_pred)
            dates.append(y.index[t])

    if len(actuals) < 6:
        continue

    results[lane] = {
        'sarimax': safe_metrics(actuals, sar_preds),
        'stacked': safe_metrics(actuals, stk_preds),
        'n_backtest': len(actuals),
    }

    bt_df = pd.DataFrame({
        'actual':       actuals,
        'sarimax_pred': sar_preds,
        'stacked_pred': stk_preds,
    }, index=pd.DatetimeIndex(dates))
    bt_df.to_csv(os.path.join(OUT, f'backtest_{lane}.csv'))
    all_bt.append({'lane': lane,
                   'sarimax_mape':    results[lane]['sarimax']['mape'],
                   'stacked_mape':    results[lane]['stacked']['mape'],
                   'stacked_dir_acc': results[lane]['stacked']['dir_acc'],
                   'n_months':        results[lane]['n_backtest']})

    print(f"  [{i+1:2d}/{len(DEMO_LANES)}] {lane:8s} | "
          f"SAR {results[lane]['sarimax']['mape']:5.1f}% | "
          f"STK {results[lane]['stacked']['mape']:5.1f}% | "
          f"Dir {results[lane]['stacked']['dir_acc']:5.1f}% | "
          f"{time.time()-tl:.0f}s", flush=True)

# ── Save backtest summary ─────────────────────────────────────────────────────
with open(os.path.join(OUT, 'backtest_summary_antarctica.json'), 'w') as f:
    json.dump(results, f, indent=2)

summary_df = pd.DataFrame(all_bt).sort_values('stacked_mape')
summary_df.to_csv(os.path.join(OUT, 'backtest_summary_table.csv'), index=False)

print(f"\n{'='*60}", flush=True)
print(f"BACKTEST COMPLETE — {len(results)} lanes | {time.time()-t0:.0f}s total", flush=True)
print(f"{'='*60}", flush=True)
print(f"Median SARIMAX MAPE : {summary_df['sarimax_mape'].median():.1f}%", flush=True)
print(f"Median Stacked MAPE : {summary_df['stacked_mape'].median():.1f}%", flush=True)
print(f"Median Dir Accuracy : {summary_df['stacked_dir_acc'].median():.1f}%", flush=True)
print(f"\nAll lanes results:", flush=True)
print(summary_df.to_string(index=False), flush=True)

# ── Train final models on full data ──────────────────────────────────────────
print(f"\nTraining final models on full data...", flush=True)
t1 = time.time()
final_models = {}

for lane in DEMO_LANES:
    y = panel[lane].dropna()
    if len(y) < MIN_TRAIN:
        continue

    exog_lane = exog.reindex(y.index)[EXOG_COLS].ffill().bfill()
    xgb_feat  = make_xgb_features(y, exog_lane)
    y_log     = np.log1p(y)

    try:
        m = SARIMAX(y_log, exog=exog_lane,
                    order=SARIMAX_ORDER, seasonal_order=SARIMAX_SORDER,
                    enforce_stationarity=False, enforce_invertibility=False)
        r = m.fit(disp=False, maxiter=100, method='lbfgs')

        insample = np.expm1(r.fittedvalues.values)
        resid    = y.values - insample

        feat_full = xgb_feat.copy()
        feat_full['sar_pred'] = insample
        feat_full = feat_full.dropna()
        resid_al  = resid[-len(feat_full):]

        xgb_final = XGBRegressor(n_estimators=80, max_depth=3,
                                 learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=0.8,
                                 random_state=42, verbosity=0)
        xgb_final.fit(feat_full, resid_al)

        fi = dict(zip(feat_full.columns,
                      xgb_final.feature_importances_.tolist()))

        final_models[lane] = {
            'sarimax_result':     r,
            'xgb_model':          xgb_final,
            'last_y':             y,
            'last_exog':          exog_lane,
            'xgb_feat_cols':      list(feat_full.columns),
            'feature_importance': fi,
        }
        print(f"  Final model trained: {lane}", flush=True)
    except Exception as e:
        print(f"  FAILED {lane}: {e}", flush=True)

with open(os.path.join(MDL, 'final_models_antarctica.pkl'), 'wb') as f:
    pickle.dump(final_models, f)

# Save demo lane list for dashboard
with open(os.path.join(DATA, 'demo_lanes.json'), 'w') as f:
    json.dump(DEMO_LANES, f, indent=2)

print(f"\nSaved {len(final_models)} final models | {time.time()-t1:.0f}s", flush=True)
print("All done.", flush=True)
