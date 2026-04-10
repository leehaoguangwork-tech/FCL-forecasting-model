"""
SARIMAX → LSTM → XGBoost Triple-Stack Training
================================================
Trains on two lanes:
  - THBKK (lowest MAPE, High reliability)
  - USLAX (highest MAPE, Low reliability)

Architecture:
  1. SARIMAX: fits linear trend + seasonality + exog macro → produces sar_pred
  2. LSTM:    takes sequence of [sar_residual, exog features] → produces lstm_correction
  3. XGBoost: takes [sar_pred, lstm_correction, exog features, lags] → final forecast

Saves to: models/antarctica/lstm_stack_{lane}.pkl
Outputs:  outputs/antarctica/testmode_{lane}_backtest.csv
          outputs/antarctica/testmode_{lane}_validation.csv
          outputs/antarctica/testmode_summary.json
"""

import os, json, warnings, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA      = os.path.join(BASE, 'data')
OUT       = os.path.join(BASE, 'outputs', 'antarctica')
MODEL_DIR = os.path.join(BASE, 'models', 'antarctica')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

TEST_LANES = ['THBKK', 'USLAX']   # lowest MAPE, highest MAPE

# ── Load data ─────────────────────────────────────────────────────────────────
panel = pd.read_parquet(os.path.join(DATA, 'antarctica_monthly_panel.parquet'))
exog  = pd.read_parquet(os.path.join(DATA, 'exog_features_antarctica.parquet'))

# Align index
panel.index = pd.to_datetime(panel.index)
exog.index  = pd.to_datetime(exog.index)

# Training window: Jul 2019 – Dec 2024 (same as Path 1 / Model with COVID)
TRAIN_END  = '2024-12-31'
VAL_START  = '2025-01-01'
TRAIN_START = '2019-07-01'

# Exog columns used in Path 1 SARIMAX (12 columns)
EXOG_COLS_P1 = [
    'brent_crude', 'usdcny', 'bdry_etf',
    'us_indpro', 'cfnai', 'china_exports',
    'dummy_ukraine', 'dummy_red_sea',
    'dummy_covid', 'dummy_supply_crunch',
    'month_sin', 'month_cos',
]

# Subset to available columns
available_exog = [c for c in EXOG_COLS_P1 if c in exog.columns]
print(f"Using {len(available_exog)} exog columns: {available_exog}")

LSTM_SEQ_LEN = 6   # 6-month lookback window for LSTM

summary = {}

for lane in TEST_LANES:
    print(f"\n{'='*60}")
    print(f"Training LSTM stack for {lane}")
    print('='*60)

    # ── 1. Prepare rate series ────────────────────────────────────
    y_full = panel[lane].dropna()
    y_train = y_full[(y_full.index >= TRAIN_START) & (y_full.index <= TRAIN_END)]
    y_val   = y_full[y_full.index >= VAL_START]

    exog_full  = exog[available_exog].reindex(y_full.index).ffill().bfill()
    exog_train = exog_full[(exog_full.index >= TRAIN_START) & (exog_full.index <= TRAIN_END)]
    exog_val   = exog_full[exog_full.index >= VAL_START]

    print(f"  Train: {y_train.index[0].date()} → {y_train.index[-1].date()} ({len(y_train)} obs)")
    print(f"  Val:   {y_val.index[0].date()} → {y_val.index[-1].date()} ({len(y_val)} obs)")

    # ── 2. Fit SARIMAX ────────────────────────────────────────────
    print("  Fitting SARIMAX...")
    try:
        sar_model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sar_res = sar_model.fit(disp=False, maxiter=200)
        print(f"  SARIMAX AIC: {sar_res.aic:.1f}")
    except Exception as e:
        print(f"  SARIMAX failed: {e}, trying simpler order")
        sar_model = SARIMAX(y_train, exog=exog_train, order=(1,1,1),
                            enforce_stationarity=False, enforce_invertibility=False)
        sar_res = sar_model.fit(disp=False, maxiter=200)

    # In-sample SARIMAX predictions on training set
    sar_train_pred = sar_res.fittedvalues
    sar_train_resid = y_train - sar_train_pred

    # ── 3. Prepare LSTM training sequences ───────────────────────
    print("  Building LSTM sequences...")
    # Features for LSTM: [sar_residual, normalised exog]
    resid_arr  = sar_train_resid.values.reshape(-1, 1)
    exog_arr   = exog_train.values

    # Normalise
    resid_scaler = MinMaxScaler()
    exog_scaler  = MinMaxScaler()
    rate_scaler  = MinMaxScaler()

    resid_norm = resid_scaler.fit_transform(resid_arr)
    exog_norm  = exog_scaler.fit_transform(exog_arr)
    rate_norm  = rate_scaler.fit_transform(y_train.values.reshape(-1, 1))

    # Combine: [normalised_residual, normalised_exog]
    lstm_features = np.hstack([resid_norm, exog_norm])  # shape (T, 1+n_exog)

    # Build sequences
    X_lstm, y_lstm = [], []
    for i in range(LSTM_SEQ_LEN, len(lstm_features)):
        X_lstm.append(lstm_features[i-LSTM_SEQ_LEN:i])
        y_lstm.append(resid_norm[i, 0])   # predict next residual

    X_lstm = np.array(X_lstm)   # (N, seq_len, features)
    y_lstm = np.array(y_lstm)   # (N,)

    print(f"  LSTM input shape: {X_lstm.shape}")

    # ── 4. Build and train LSTM ───────────────────────────────────
    print("  Training LSTM...")
    tf.random.set_seed(42)
    n_features = X_lstm.shape[2]

    lstm_model = Sequential([
        LSTM(32, input_shape=(LSTM_SEQ_LEN, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = lstm_model.fit(
        X_lstm, y_lstm,
        epochs=200,
        batch_size=8,
        validation_split=0.15,
        callbacks=[es],
        verbose=0
    )
    print(f"  LSTM trained for {len(history.history['loss'])} epochs")
    print(f"  Final train loss: {history.history['loss'][-1]:.6f}")

    # ── 5. Generate LSTM corrections on training set ──────────────
    lstm_train_corr_norm = lstm_model.predict(X_lstm, verbose=0).flatten()
    # Inverse transform to rate scale
    lstm_train_corr = resid_scaler.inverse_transform(
        lstm_train_corr_norm.reshape(-1, 1)).flatten()

    # Align: LSTM starts at index LSTM_SEQ_LEN
    lstm_start_idx = LSTM_SEQ_LEN
    sar_train_pred_aligned = sar_train_pred.iloc[lstm_start_idx:]
    y_train_aligned = y_train.iloc[lstm_start_idx:]
    exog_train_aligned = exog_train.iloc[lstm_start_idx:]

    # ── 6. Build XGBoost features ─────────────────────────────────
    print("  Building XGBoost features...")
    def build_xgb_features(y_series, sar_preds, lstm_corr, exog_df, y_full_ref):
        """Build XGBoost feature matrix."""
        rows = []
        for i in range(len(y_series)):
            idx = y_series.index[i]
            row = {}
            # SARIMAX prediction
            row['sar_pred'] = sar_preds.iloc[i] if hasattr(sar_preds, 'iloc') else sar_preds[i]
            # LSTM correction
            row['lstm_corr'] = lstm_corr[i]
            # Exog features
            for col in exog_df.columns:
                row[col] = exog_df[col].iloc[i]
            # Rate lags (from full series)
            for lag in [1, 2, 3]:
                lag_idx = y_full_ref.index.get_loc(idx) - lag
                row[f'rate_lag{lag}'] = y_full_ref.iloc[lag_idx] if lag_idx >= 0 else np.nan
            # Rolling stats
            lag1_idx = y_full_ref.index.get_loc(idx) - 1
            if lag1_idx >= 3:
                row['rate_roll3_mean'] = y_full_ref.iloc[lag1_idx-2:lag1_idx+1].mean()
                row['rate_roll3_std']  = y_full_ref.iloc[lag1_idx-2:lag1_idx+1].std()
            else:
                row['rate_roll3_mean'] = np.nan
                row['rate_roll3_std']  = np.nan
            rows.append(row)
        return pd.DataFrame(rows, index=y_series.index)

    X_xgb_train = build_xgb_features(
        y_train_aligned, sar_train_pred_aligned,
        lstm_train_corr, exog_train_aligned, y_full
    )
    y_xgb_train = y_train_aligned - sar_train_pred_aligned  # XGBoost corrects SARIMAX+LSTM residual

    # Drop NaN rows
    mask = X_xgb_train.notna().all(axis=1)
    X_xgb_train = X_xgb_train[mask]
    y_xgb_train = y_xgb_train[mask]

    # ── 7. Train XGBoost ──────────────────────────────────────────
    print("  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_xgb_train, y_xgb_train)

    # ── 8. Walk-forward backtest ──────────────────────────────────
    print("  Running walk-forward backtest...")
    bt_results = []
    # Use last LSTM_SEQ_LEN training residuals as seed for rolling window
    resid_window = list(sar_train_resid.values[-LSTM_SEQ_LEN:])
    exog_window  = list(exog_train.values[-LSTM_SEQ_LEN:])

    # Backtest on training set (last 24 months)
    bt_start = max(0, len(y_train_aligned) - 24)
    for i in range(bt_start, len(y_train_aligned)):
        idx = y_train_aligned.index[i]
        actual = y_train_aligned.iloc[i]
        sar_p  = sar_train_pred_aligned.iloc[i]
        resid  = actual - sar_p

        # LSTM prediction
        seq_resid = np.array(resid_window[-LSTM_SEQ_LEN:]).reshape(-1, 1)
        seq_exog  = np.array(exog_window[-LSTM_SEQ_LEN:])
        seq_resid_norm = resid_scaler.transform(seq_resid)
        seq_exog_norm  = exog_scaler.transform(seq_exog)
        seq = np.hstack([seq_resid_norm, seq_exog_norm]).reshape(1, LSTM_SEQ_LEN, n_features)
        lstm_c_norm = lstm_model.predict(seq, verbose=0)[0, 0]
        lstm_c = resid_scaler.inverse_transform([[lstm_c_norm]])[0, 0]

        # XGBoost features
        xf = {'sar_pred': sar_p, 'lstm_corr': lstm_c}
        for col in exog_train_aligned.columns:
            xf[col] = exog_train_aligned[col].iloc[i]
        for lag in [1, 2, 3]:
            lag_pos = y_full.index.get_loc(idx) - lag
            xf[f'rate_lag{lag}'] = y_full.iloc[lag_pos] if lag_pos >= 0 else np.nan
        lag1_pos = y_full.index.get_loc(idx) - 1
        if lag1_pos >= 3:
            xf['rate_roll3_mean'] = y_full.iloc[lag1_pos-2:lag1_pos+1].mean()
            xf['rate_roll3_std']  = y_full.iloc[lag1_pos-2:lag1_pos+1].std()
        else:
            xf['rate_roll3_mean'] = np.nan
            xf['rate_roll3_std']  = np.nan

        xf_df = pd.DataFrame([xf])
        xf_df = xf_df.reindex(columns=X_xgb_train.columns, fill_value=0)
        xgb_c = xgb_model.predict(xf_df)[0]

        # Cap XGBoost correction at ±30% of SARIMAX
        cap = abs(sar_p) * 0.30
        xgb_c = np.clip(xgb_c, -cap, cap)

        final = sar_p + xgb_c

        bt_results.append({
            'date': idx, 'actual': actual,
            'sar_pred': sar_p, 'lstm_corr': lstm_c,
            'xgb_corr': xgb_c, 'triple_stack': final
        })

        # Update rolling window
        resid_window.append(resid)
        exog_window.append(exog_train.loc[idx].values)

    bt_df = pd.DataFrame(bt_results).set_index('date')

    # ── 9. Validation (Jan 2025 – Apr 2026) ──────────────────────
    print("  Running validation...")
    val_results = []
    # Seed with last LSTM_SEQ_LEN training residuals
    resid_window = list(sar_train_resid.values[-LSTM_SEQ_LEN:])
    exog_window  = list(exog_train.values[-LSTM_SEQ_LEN:])

    # Get SARIMAX validation predictions
    try:
        sar_val_pred = sar_res.forecast(steps=len(y_val), exog=exog_val)
    except Exception:
        sar_val_pred = sar_res.predict(
            start=y_val.index[0], end=y_val.index[-1], exog=exog_val)

    for i in range(len(y_val)):
        idx    = y_val.index[i]
        actual = y_val.iloc[i]
        sar_p  = sar_val_pred.iloc[i]

        # LSTM
        seq_resid = np.array(resid_window[-LSTM_SEQ_LEN:]).reshape(-1, 1)
        seq_exog  = np.array(exog_window[-LSTM_SEQ_LEN:])
        seq_resid_norm = resid_scaler.transform(seq_resid)
        seq_exog_norm  = exog_scaler.transform(seq_exog)
        seq = np.hstack([seq_resid_norm, seq_exog_norm]).reshape(1, LSTM_SEQ_LEN, n_features)
        lstm_c_norm = lstm_model.predict(seq, verbose=0)[0, 0]
        lstm_c = resid_scaler.inverse_transform([[lstm_c_norm]])[0, 0]

        # XGBoost
        xf = {'sar_pred': sar_p, 'lstm_corr': lstm_c}
        for col in exog_val.columns:
            xf[col] = exog_val[col].iloc[i]
        for lag in [1, 2, 3]:
            lag_pos = y_full.index.get_loc(idx) - lag if idx in y_full.index else -1
            if lag_pos >= 0:
                xf[f'rate_lag{lag}'] = y_full.iloc[lag_pos]
            else:
                xf[f'rate_lag{lag}'] = y_train.iloc[-lag] if lag <= len(y_train) else np.nan
        lag1_pos = y_full.index.get_loc(idx) - 1 if idx in y_full.index else -1
        if lag1_pos >= 3:
            xf['rate_roll3_mean'] = y_full.iloc[lag1_pos-2:lag1_pos+1].mean()
            xf['rate_roll3_std']  = y_full.iloc[lag1_pos-2:lag1_pos+1].std()
        else:
            xf['rate_roll3_mean'] = y_train.iloc[-3:].mean()
            xf['rate_roll3_std']  = y_train.iloc[-3:].std()

        xf_df = pd.DataFrame([xf])
        xf_df = xf_df.reindex(columns=X_xgb_train.columns, fill_value=0)
        xgb_c = xgb_model.predict(xf_df)[0]
        cap   = abs(sar_p) * 0.30
        xgb_c = np.clip(xgb_c, -cap, cap)
        final = sar_p + xgb_c

        val_results.append({
            'date': idx, 'actual': actual,
            'sar_pred': sar_p, 'lstm_corr': lstm_c,
            'xgb_corr': xgb_c, 'triple_stack': final
        })

        # Update rolling window with predicted residual (no actual available in future)
        pred_resid = actual - sar_p
        resid_window.append(pred_resid)
        exog_window.append(exog_val.iloc[i].values)

    val_df = pd.DataFrame(val_results).set_index('date')

    # ── 10. Compute MAPEs ─────────────────────────────────────────
    sar_mape  = mean_absolute_percentage_error(val_df['actual'], val_df['sar_pred']) * 100
    trip_mape = mean_absolute_percentage_error(val_df['actual'], val_df['triple_stack']) * 100

    # Load baseline (Path 1) MAPE for comparison
    v1 = pd.read_csv(os.path.join(OUT, 'path1_val_table.csv'))
    baseline_sar_mape = v1.loc[v1['lane']==lane, 'val_sar_mape'].values[0]
    baseline_stk_mape = v1.loc[v1['lane']==lane, 'val_stk_mape'].values[0]

    print(f"\n  Results for {lane}:")
    print(f"    Baseline SARIMAX MAPE:        {baseline_sar_mape:.2f}%")
    print(f"    Baseline Stacked MAPE:        {baseline_stk_mape:.2f}%")
    print(f"    Triple-Stack SARIMAX MAPE:    {sar_mape:.2f}%")
    print(f"    Triple-Stack Final MAPE:      {trip_mape:.2f}%")
    print(f"    Improvement vs baseline stk:  {baseline_stk_mape - trip_mape:+.2f}pp")

    summary[lane] = {
        'baseline_sar_mape':  round(baseline_sar_mape, 2),
        'baseline_stk_mape':  round(baseline_stk_mape, 2),
        'triple_sar_mape':    round(sar_mape, 2),
        'triple_final_mape':  round(trip_mape, 2),
        'improvement_pp':     round(baseline_stk_mape - trip_mape, 2),
        'n_train':            int(len(y_train)),
        'n_val':              int(len(y_val)),
        'lstm_epochs':        int(len(history.history['loss'])),
    }

    # ── 11. Save outputs ──────────────────────────────────────────
    bt_df.to_csv(os.path.join(OUT, f'testmode_{lane}_backtest.csv'))
    val_df.to_csv(os.path.join(OUT, f'testmode_{lane}_validation.csv'))

    # Save model bundle
    bundle = {
        'sar_res':       sar_res,
        'lstm_model':    lstm_model,
        'xgb_model':     xgb_model,
        'resid_scaler':  resid_scaler,
        'exog_scaler':   exog_scaler,
        'rate_scaler':   rate_scaler,
        'xgb_feat_cols': list(X_xgb_train.columns),
        'exog_cols':     available_exog,
        'seq_len':       LSTM_SEQ_LEN,
        'n_features':    n_features,
        'last_resid_window': resid_window[-LSTM_SEQ_LEN:],
        'last_exog_window':  exog_window[-LSTM_SEQ_LEN:],
        'last_y':        y_train.iloc[-1],
    }
    with open(os.path.join(MODEL_DIR, f'lstm_stack_{lane}.pkl'), 'wb') as f:
        pickle.dump(bundle, f)

    print(f"  Saved model bundle: lstm_stack_{lane}.pkl")
    print(f"  Saved backtest:     testmode_{lane}_backtest.csv")
    print(f"  Saved validation:   testmode_{lane}_validation.csv")

# Save summary
with open(os.path.join(OUT, 'testmode_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(json.dumps(summary, indent=2))
