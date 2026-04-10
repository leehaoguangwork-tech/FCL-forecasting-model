"""
PATH 1 (A+C): Full data (Jul 2019–Dec 2024) + Shock Dummies + Reliability Score
=================================================================================
- SARIMAX with explicit shock dummies (COVID spike, post-spike crash, Red Sea)
- XGBoost residual correction CAPPED at ±30% of SARIMAX prediction
- Reliability score (High/Medium/Low) based on CV and backtest MAPE
- Walk-forward backtest on full training window
- Jan 2025–Apr 2026 out-of-sample validation (using panel data already collected)
"""

import os, json, warnings, pickle, time
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

DEMO_LANES = ['CNSHA','JPTYO','KRPUS','INNSA','THBKK',
              'DEHAM','NLRTM','GBFXT','USNYC','USLAX',
              'ARBUE','AEJEA','NGLOS','AUSYD','QAHMD']

panel = pd.read_parquet(os.path.join(DATA, 'antarctica_monthly_panel.parquet'))
exog  = pd.read_parquet(os.path.join(DATA, 'exog_features_antarctica.parquet'))

EXOG_COLS = ['brent_crude','usdcny','us_indpro','us_cfnai','bdry_etf',
             'dummy_covid','dummy_supply_crunch','dummy_ukraine',
             'dummy_red_sea','dummy_hormuz']

# Extra shock dummies for Path 1
def add_shock_dummies(idx):
    """Add COVID spike and post-spike crash dummies."""
    d = pd.DataFrame(index=idx)
    d['shock_covid_spike']  = ((idx >= '2021-06-01') & (idx <= '2022-06-30')).astype(float)
    d['shock_post_crash']   = ((idx >= '2022-07-01') & (idx <= '2023-03-31')).astype(float)
    return d

SARIMAX_ORDER  = (1, 1, 1)
SARIMAX_SORDER = (0, 1, 1, 12)
MIN_TRAIN = 24
XGB_CAP   = 0.30   # cap XGBoost correction at ±30% of SARIMAX prediction

def make_xgb_features(y_series, exog_df, shock_df):
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
    for col in shock_df.columns:
        df[col] = shock_df[col].values
    return df

def safe_metrics(act, pred):
    act  = np.array(act,  dtype=float)
    pred = np.array(pred, dtype=float)
    mask = np.isfinite(act) & np.isfinite(pred) & (act > 0)
    act  = act[mask]; pred = pred[mask]
    if len(act) < 2:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'dir_acc': np.nan}
    pred = np.clip(pred, 0, act.max() * 5)
    mae  = float(mean_absolute_error(act, pred))
    rmse = float(np.sqrt(mean_squared_error(act, pred)))
    mape = float(np.mean(np.abs((act - pred) / np.maximum(act, 1))) * 100)
    da   = float(np.mean(np.sign(np.diff(act)) == np.sign(np.diff(pred))) * 100) if len(act) > 1 else np.nan
    return {'mae': round(mae,2), 'rmse': round(rmse,2),
            'mape': round(mape,2), 'dir_acc': round(da,2)}

def reliability_score(cv, mape):
    """Assign reliability tier based on coefficient of variation and backtest MAPE."""
    if cv <= 0.40 and mape <= 25:
        return 'High'
    elif cv <= 0.60 and mape <= 50:
        return 'Medium'
    else:
        return 'Low'

# ── Walk-forward backtest ─────────────────────────────────────────────────────
results = {}
all_bt  = []
print(f"PATH 1 (A+C): Full data + shock dummies | {len(DEMO_LANES)} lanes", flush=True)
print("="*65, flush=True)
t0 = time.time()

for i, lane in enumerate(DEMO_LANES):
    tl = time.time()
    y = panel[lane].dropna()
    # Use all data up to Dec 2024 for training/backtest
    y_train = y[y.index < '2025-01-01']
    y_val   = y[y.index >= '2025-01-01']   # Jan 2025–Apr 2026 validation

    if len(y_train) < MIN_TRAIN + 6:
        print(f"  [{i+1:2d}/{len(DEMO_LANES)}] {lane}: SKIP", flush=True)
        continue

    exog_full  = exog.reindex(y_train.index)[EXOG_COLS].ffill().bfill()
    shock_full = add_shock_dummies(y_train.index)
    exog_full  = pd.concat([exog_full, shock_full], axis=1)
    exog_cols_all = list(exog_full.columns)

    xgb_feat   = make_xgb_features(y_train, exog_full, shock_full)
    y_log      = np.log1p(y_train)

    cv = float(y_train.std() / y_train.mean())

    # Fit initial SARIMAX on first MIN_TRAIN months
    try:
        m0 = SARIMAX(y_log.iloc[:MIN_TRAIN],
                     exog=exog_full.iloc[:MIN_TRAIN],
                     order=SARIMAX_ORDER,
                     seasonal_order=SARIMAX_SORDER,
                     enforce_stationarity=False,
                     enforce_invertibility=False)
        r0 = m0.fit(disp=False, maxiter=100, method='lbfgs')
    except Exception as e:
        print(f"  [{i+1:2d}] {lane}: SARIMAX init FAILED: {e}", flush=True)
        continue

    actuals, sar_preds, stk_preds, dates = [], [], [], []
    xgb_model = None
    xgb_last_trained = -999

    for t in range(MIN_TRAIN, len(y_train)):
        try:
            r_ext    = r0.apply(y_log.iloc[:t], exog=exog_full.iloc[:t], refit=False)
            ex_te    = exog_full.iloc[[t]]
            fc_log   = float(r_ext.forecast(steps=1, exog=ex_te).iloc[0])
            sar_pred = float(np.expm1(fc_log))
            insample = np.expm1(r_ext.fittedvalues.values)
            resid    = y_train.iloc[:t].values - insample
        except Exception:
            continue

        if t - xgb_last_trained >= 6 and t >= MIN_TRAIN + 6:
            try:
                feat_tr = xgb_feat.iloc[:t].copy()
                feat_tr['sar_pred'] = insample
                feat_tr = feat_tr.dropna()
                resid_al = resid[-len(feat_tr):]
                if len(feat_tr) >= 12:
                    xgb_m = XGBRegressor(n_estimators=60, max_depth=3,
                                         learning_rate=0.1, subsample=0.8,
                                         colsample_bytree=0.8,
                                         random_state=42, verbosity=0)
                    xgb_m.fit(feat_tr, resid_al)
                    xgb_model = xgb_m
                    xgb_last_trained = t
            except Exception:
                xgb_model = None

        if xgb_model is not None:
            try:
                feat_te = xgb_feat.iloc[[t]].copy()
                feat_te['sar_pred'] = sar_pred
                feat_te = feat_te.fillna(0)
                corr = float(xgb_model.predict(feat_te)[0])
                # CAP correction at ±30% of SARIMAX prediction
                cap = XGB_CAP * abs(sar_pred)
                corr = np.clip(corr, -cap, cap)
                stk_pred = sar_pred + corr
            except Exception:
                stk_pred = sar_pred
        else:
            stk_pred = sar_pred

        if np.isfinite(sar_pred) and np.isfinite(stk_pred):
            actuals.append(float(y_train.iloc[t]))
            sar_preds.append(sar_pred)
            stk_preds.append(stk_pred)
            dates.append(y_train.index[t])

    if len(actuals) < 6:
        continue

    sar_m = safe_metrics(actuals, sar_preds)
    stk_m = safe_metrics(actuals, stk_preds)
    rel   = reliability_score(cv, min(sar_m['mape'], stk_m['mape']))

    # ── Jan 2025–Apr 2026 out-of-sample validation ───────────────────────────
    val_actuals, val_sar, val_stk, val_dates = [], [], [], []
    if len(y_val) > 0:
        # Refit final SARIMAX on full training data
        try:
            m_final = SARIMAX(y_log,
                              exog=exog_full,
                              order=SARIMAX_ORDER,
                              seasonal_order=SARIMAX_SORDER,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
            r_final = m_final.fit(disp=False, maxiter=100, method='lbfgs')

            # Retrain XGBoost on full training data
            feat_full = xgb_feat.copy()
            insample_full = np.expm1(r_final.fittedvalues.values)
            feat_full['sar_pred'] = insample_full
            feat_full = feat_full.dropna()
            resid_full = y_train.values[-len(feat_full):] - insample_full[-len(feat_full):]
            xgb_final = XGBRegressor(n_estimators=60, max_depth=3,
                                     learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8,
                                     random_state=42, verbosity=0)
            xgb_final.fit(feat_full, resid_full)

            # Forecast each validation month using rolling actual
            y_rolling = y_train.copy()
            for vt in range(len(y_val)):
                y_r_log = np.log1p(y_rolling)
                exog_r  = exog.reindex(y_rolling.index)[EXOG_COLS].ffill().bfill()
                shock_r = add_shock_dummies(y_rolling.index)
                exog_r  = pd.concat([exog_r, shock_r], axis=1)

                try:
                    r_v = r_final.apply(y_r_log, exog=exog_r, refit=False)
                    # Get exog for the forecast step
                    val_idx  = y_val.index[vt]
                    exog_vt  = exog.reindex([val_idx])[EXOG_COLS].ffill().bfill()
                    shock_vt = add_shock_dummies(pd.DatetimeIndex([val_idx]))
                    exog_vt  = pd.concat([exog_vt, shock_vt], axis=1)
                    fc_log_v = float(r_v.forecast(steps=1, exog=exog_vt).iloc[0])
                    sp_v     = float(np.expm1(fc_log_v))

                    # XGBoost correction
                    xf = make_xgb_features(
                        pd.concat([y_rolling, y_val.iloc[:vt+1]]),
                        pd.concat([exog_r, exog_vt]),
                        pd.concat([shock_r, shock_vt])
                    ).iloc[[-1]].copy()
                    xf['sar_pred'] = sp_v
                    xf = xf.fillna(0)
                    corr_v = float(xgb_final.predict(xf)[0])
                    cap_v  = XGB_CAP * abs(sp_v)
                    corr_v = np.clip(corr_v, -cap_v, cap_v)
                    stk_v  = sp_v + corr_v

                    val_actuals.append(float(y_val.iloc[vt]))
                    val_sar.append(sp_v)
                    val_stk.append(stk_v)
                    val_dates.append(val_idx)

                    # Roll forward with actual
                    y_rolling = pd.concat([y_rolling, y_val.iloc[[vt]]])
                except Exception:
                    pass
        except Exception as e:
            pass

    val_sar_m = safe_metrics(val_actuals, val_sar) if val_actuals else {}
    val_stk_m = safe_metrics(val_actuals, val_stk) if val_actuals else {}

    results[lane] = {
        'cv': round(cv, 3),
        'reliability': rel,
        'backtest': {'sarimax': sar_m, 'stacked': stk_m, 'n': len(actuals)},
        'validation_2025': {
            'sarimax': val_sar_m,
            'stacked': val_stk_m,
            'n': len(val_actuals)
        }
    }

    # Save backtest CSV
    bt_df = pd.DataFrame({'actual': actuals, 'sarimax_pred': sar_preds,
                          'stacked_pred': stk_preds}, index=pd.DatetimeIndex(dates))
    bt_df.to_csv(os.path.join(OUT, f'path1_backtest_{lane}.csv'))

    # Save validation CSV
    if val_actuals:
        vl_df = pd.DataFrame({'actual': val_actuals, 'sarimax_pred': val_sar,
                              'stacked_pred': val_stk}, index=pd.DatetimeIndex(val_dates))
        vl_df.to_csv(os.path.join(OUT, f'path1_validation_{lane}.csv'))

    all_bt.append({
        'lane': lane, 'reliability': rel, 'cv': round(cv,3),
        'bt_sar_mape':  sar_m['mape'],  'bt_stk_mape':  stk_m['mape'],
        'val_sar_mape': val_sar_m.get('mape', np.nan),
        'val_stk_mape': val_stk_m.get('mape', np.nan),
        'val_n': len(val_actuals)
    })

    print(f"  [{i+1:2d}/{len(DEMO_LANES)}] {lane:6s} [{rel:6s}] "
          f"BT-SAR {sar_m['mape']:5.1f}% BT-STK {stk_m['mape']:5.1f}% | "
          f"VAL-SAR {val_sar_m.get('mape', float('nan')):5.1f}% "
          f"VAL-STK {val_stk_m.get('mape', float('nan')):5.1f}% | "
          f"{time.time()-tl:.0f}s", flush=True)

# ── Save results ──────────────────────────────────────────────────────────────
with open(os.path.join(OUT, 'path1_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

summary = pd.DataFrame(all_bt).sort_values('val_sar_mape')
summary.to_csv(os.path.join(OUT, 'path1_summary.csv'), index=False)

print(f"\n{'='*65}", flush=True)
print(f"PATH 1 COMPLETE — {len(results)} lanes | {time.time()-t0:.0f}s", flush=True)
print(f"{'='*65}", flush=True)
print(f"Median BT SARIMAX MAPE  : {summary['bt_sar_mape'].median():.1f}%", flush=True)
print(f"Median BT Stacked MAPE  : {summary['bt_stk_mape'].median():.1f}%", flush=True)
print(f"Median VAL SARIMAX MAPE : {summary['val_sar_mape'].median():.1f}%", flush=True)
print(f"Median VAL Stacked MAPE : {summary['val_stk_mape'].median():.1f}%", flush=True)
print(f"\n{summary.to_string(index=False)}", flush=True)

# ── Train final models on full data (up to Apr 2026) ─────────────────────────
print(f"\nTraining final models on full data (up to Apr 2026)...", flush=True)
final_models = {}

for lane in DEMO_LANES:
    y = panel[lane].dropna()
    if len(y) < MIN_TRAIN:
        continue
    exog_lane  = exog.reindex(y.index)[EXOG_COLS].ffill().bfill()
    shock_lane = add_shock_dummies(y.index)
    exog_all   = pd.concat([exog_lane, shock_lane], axis=1)
    xgb_feat   = make_xgb_features(y, exog_all, shock_lane)
    y_log      = np.log1p(y)

    try:
        m = SARIMAX(y_log, exog=exog_all,
                    order=SARIMAX_ORDER, seasonal_order=SARIMAX_SORDER,
                    enforce_stationarity=False, enforce_invertibility=False)
        r = m.fit(disp=False, maxiter=100, method='lbfgs')
        insample = np.expm1(r.fittedvalues.values)
        resid    = y.values - insample
        feat_f   = xgb_feat.copy()
        feat_f['sar_pred'] = insample
        feat_f   = feat_f.dropna()
        resid_f  = resid[-len(feat_f):]
        xgb_f    = XGBRegressor(n_estimators=60, max_depth=3,
                                learning_rate=0.1, subsample=0.8,
                                colsample_bytree=0.8, random_state=42, verbosity=0)
        xgb_f.fit(feat_f, resid_f)
        fi = dict(zip(feat_f.columns, xgb_f.feature_importances_.tolist()))
        final_models[lane] = {
            'sarimax_result': r, 'xgb_model': xgb_f,
            'last_y': y, 'last_exog': exog_all,
            'xgb_feat_cols': list(feat_f.columns),
            'feature_importance': fi,
            'reliability': results.get(lane, {}).get('reliability', 'Unknown'),
            'cv': results.get(lane, {}).get('cv', np.nan),
            'path': 'A+C'
        }
        print(f"  {lane}: trained", flush=True)
    except Exception as e:
        print(f"  {lane}: FAILED — {e}", flush=True)

with open(os.path.join(MDL, 'path1_final_models.pkl'), 'wb') as f:
    pickle.dump(final_models, f)

print(f"\nSaved {len(final_models)} Path 1 final models.", flush=True)
print("PATH 1 DONE.", flush=True)
