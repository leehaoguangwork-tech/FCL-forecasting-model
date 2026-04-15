"""
PATH 2 v2.0: Stable data (Jul 2021–Apr 2026, excluding COVID spike) + Lane-Specific FX + Maersk proxy
=======================================================================================================
Changes from v1.0:
  - USD/CNY replaced by lane-specific currency pair per destination region
  - Maersk B share price added as container freight index proxy
  - AED lanes (AEJEA, QAHMD) and USD lanes (USLAX, USNYC) use no FX variable
  - XGB_CAP reduced to 0.10 (from 0.30) per grid search recommendation
  - All other architecture identical to v1.0 for fair comparison
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

LANE_FX = {
    'CNSHA': 'usd_cny', 'JPTYO': 'usd_cny', 'KRPUS': 'usd_cny',
    'INNSA': 'usd_inr', 'THBKK': 'usd_cny',
    'DEHAM': 'usd_eur', 'NLRTM': 'usd_eur', 'GBFXT': 'usd_eur',
    'USNYC': None, 'USLAX': None,
    'ARBUE': 'usd_ars', 'AEJEA': None, 'NGLOS': 'usd_ngn',
    'AUSYD': 'usd_aud', 'QAHMD': None,
}
DEMO_LANES = list(LANE_FX.keys())

# Note: maersk_proxy excluded from Path 2 — shorter stable window (42 months) causes
# multicollinearity and numerical instability with the additional feature.
# Maersk proxy is included in Path 1 (full 65-month window) only.
BASE_EXOG_COLS = ['brent_crude', 'us_indpro', 'us_cfnai', 'bdry_etf',
                  'dummy_ukraine', 'dummy_red_sea', 'dummy_hormuz']

SARIMAX_ORDER  = (1, 1, 1)
SARIMAX_SORDER = (0, 1, 1, 12)
MIN_TRAIN = 24
XGB_CAP   = 0.10   # Tightened per grid search recommendation

def get_lane_exog_cols(lane):
    fx = LANE_FX.get(lane)
    cols = BASE_EXOG_COLS.copy()
    if fx:
        cols.append(fx)
    return cols

def make_xgb_features(y_series, exog_df, exog_cols):
    df = pd.DataFrame(index=y_series.index)
    for lag in [1, 2, 3, 6, 12]:
        df[f'rate_lag{lag}'] = y_series.shift(lag)
    df['rate_roll3'] = y_series.shift(1).rolling(3).mean()
    df['rate_roll6'] = y_series.shift(1).rolling(6).mean()
    df['month_sin']  = np.sin(2 * np.pi * y_series.index.month / 12)
    df['month_cos']  = np.cos(2 * np.pi * y_series.index.month / 12)
    for col in exog_cols:
        if col in exog_df.columns:
            df[col]         = exog_df[col].values
            df[f'{col}_l1'] = exog_df[col].shift(1).values
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
    if cv <= 0.40 and mape <= 25:
        return 'High'
    elif cv <= 0.60 and mape <= 50:
        return 'Medium'
    else:
        return 'Low'

# ── Load data ─────────────────────────────────────────────────────────────────
panel    = pd.read_parquet(os.path.join(DATA, 'antarctica_monthly_panel.parquet'))
exog_v1  = pd.read_parquet(os.path.join(DATA, 'exog_features_antarctica.parquet')).rename(columns={'usdcny': 'usd_cny'})
new_exog = pd.read_parquet(os.path.join(DATA, 'new_exog_v2.parquet'))
exog_all = exog_v1.copy()
for col in new_exog.columns:
    exog_all[col] = new_exog[col].reindex(exog_all.index)
exog_all['maersk_proxy'] = exog_all['maersk_proxy'] / 1000.0
exog_all = exog_all.ffill().bfill()

# Stable period: exclude COVID spike (Jul 2019–Jun 2021) — start from Jul 2021
STABLE_START = '2021-07-01'

results = {}
all_bt  = []
print(f"PATH 2 v2.0: Stable data (from {STABLE_START}) + Lane-specific FX + Maersk proxy | {len(DEMO_LANES)} lanes", flush=True)
print("="*75, flush=True)
t0 = time.time()

for i, lane in enumerate(DEMO_LANES):
    tl = time.time()
    y = panel[lane].dropna()
    y = y[y.index >= STABLE_START]  # Exclude COVID spike period
    y_train = y[y.index < '2025-01-01']
    y_val   = y[y.index >= '2025-01-01']

    if len(y_train) < MIN_TRAIN + 6:
        print(f"  [{i+1:2d}/{len(DEMO_LANES)}] {lane}: SKIP (insufficient data: {len(y_train)})", flush=True)
        continue

    exog_cols = get_lane_exog_cols(lane)
    exog_cols = [c for c in exog_cols if c in exog_all.columns]

    exog_full = exog_all.reindex(y_train.index)[exog_cols].ffill().bfill()
    xgb_feat  = make_xgb_features(y_train, exog_full, exog_cols)
    y_log     = np.log1p(y_train)
    cv        = float(y_train.std() / y_train.mean())

    actuals, sar_preds, stk_preds, dates = [], [], [], []
    xgb_model = None
    xgb_last_trained = MIN_TRAIN

    for t in range(MIN_TRAIN, len(y_train)):
        ex_tr = exog_full.iloc[:t]
        ex_te = exog_full.iloc[[t]]
        try:
            m_wf = SARIMAX(y_log.iloc[:t], exog=ex_tr,
                           order=SARIMAX_ORDER, seasonal_order=SARIMAX_SORDER,
                           enforce_stationarity=False, enforce_invertibility=False)
            r_wf = m_wf.fit(disp=False, maxiter=80, method='lbfgs')
            fc_log   = float(r_wf.forecast(steps=1, exog=ex_te).iloc[0])
            sar_pred = float(np.expm1(fc_log))
            insample = np.expm1(r_wf.fittedvalues.values)
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
                                         colsample_bytree=0.8, random_state=42, verbosity=0)
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
                cap  = XGB_CAP * abs(sar_pred)
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

    # ── Validation ────────────────────────────────────────────────────────────
    val_actuals, val_sar, val_stk, val_dates = [], [], [], []
    val_sar_m = {}; val_stk_m = {}

    if len(y_val) > 0:
        try:
            m_final = SARIMAX(y_log, exog=exog_full,
                              order=SARIMAX_ORDER, seasonal_order=SARIMAX_SORDER,
                              enforce_stationarity=False, enforce_invertibility=False)
            r_final = m_final.fit(disp=False, maxiter=100, method='lbfgs')
            insample_full = np.expm1(r_final.fittedvalues.values)
            feat_full = xgb_feat.copy()
            feat_full['sar_pred'] = insample_full
            feat_full = feat_full.dropna()
            resid_full = y_train.values[-len(feat_full):] - insample_full[-len(feat_full):]
            xgb_final = XGBRegressor(n_estimators=60, max_depth=3,
                                     learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, random_state=42, verbosity=0)
            xgb_final.fit(feat_full, resid_full)

            y_rolling = y_train.copy()
            for vt in range(len(y_val)):
                y_r_log = np.log1p(y_rolling)
                exog_r  = exog_all.reindex(y_rolling.index)[exog_cols].ffill().bfill()
                try:
                    r_v = r_final.apply(y_r_log, exog=exog_r, refit=False)
                    val_idx = y_val.index[vt]
                    exog_vt = exog_all.reindex([val_idx])[exog_cols].ffill().bfill()
                    fc_log_v = float(r_v.forecast(steps=1, exog=exog_vt).iloc[0])
                    sp_v     = float(np.expm1(fc_log_v))
                    y_ext    = pd.concat([y_rolling, y_val.iloc[:vt+1]])
                    exog_ext = pd.concat([exog_r, exog_vt])
                    xf = make_xgb_features(y_ext, exog_ext, exog_cols).iloc[[-1]].copy()
                    xf['sar_pred'] = sp_v
                    for c in feat_full.columns:
                        if c not in xf.columns:
                            xf[c] = 0.0
                    xf = xf[feat_full.columns].fillna(0)
                    corr_v = float(xgb_final.predict(xf)[0])
                    cap_v  = XGB_CAP * abs(sp_v)
                    corr_v = np.clip(corr_v, -cap_v, cap_v)
                    stk_v  = sp_v + corr_v
                    val_actuals.append(float(y_val.iloc[vt]))
                    val_sar.append(sp_v)
                    val_stk.append(stk_v)
                    val_dates.append(val_idx)
                    y_rolling = pd.concat([y_rolling, y_val.iloc[vt:vt+1]])
                except Exception as e:
                    continue
        except Exception as e:
            pass

    if val_actuals:
        val_sar_m = safe_metrics(val_actuals, val_sar)
        val_stk_m = safe_metrics(val_actuals, val_stk)
        vl_df = pd.DataFrame({
            'actual': val_actuals, 'sarimax_pred': val_sar, 'stacked_pred': val_stk
        }, index=val_dates)
        vl_df.to_csv(os.path.join(OUT, f'path2v2_validation_{lane}.csv'))

    results[lane] = {
        'reliability': rel, 'cv': round(cv, 3),
        'bt_sar_mape': sar_m['mape'], 'bt_stk_mape': stk_m['mape'],
        'val_sar_mape': val_sar_m.get('mape', np.nan),
        'val_stk_mape': val_stk_m.get('mape', np.nan),
        'fx_used': LANE_FX.get(lane, 'none'),
        'exog_cols': exog_cols,
    }
    all_bt.append({
        'lane': lane, 'reliability': rel, 'cv': round(cv, 3),
        'bt_sar_mape': sar_m['mape'], 'bt_stk_mape': stk_m['mape'],
        'val_sar_mape': val_sar_m.get('mape', np.nan),
        'val_stk_mape': val_stk_m.get('mape', np.nan),
        'fx_used': LANE_FX.get(lane, 'none'),
    })
    print(f"  [{i+1:2d}/{len(DEMO_LANES)}] {lane:6s} [{rel:6s}] FX={LANE_FX.get(lane,'none') or 'none':8s} "
          f"BT-SAR {sar_m['mape']:5.1f}% BT-STK {stk_m['mape']:5.1f}% | "
          f"VAL-SAR {val_sar_m.get('mape', float('nan')):5.1f}% "
          f"VAL-STK {val_stk_m.get('mape', float('nan')):5.1f}% | "
          f"{time.time()-tl:.0f}s", flush=True)

# ── Save results ──────────────────────────────────────────────────────────────
with open(os.path.join(OUT, 'path2v2_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
summary = pd.DataFrame(all_bt).sort_values('val_sar_mape')
summary.to_csv(os.path.join(OUT, 'path2v2_summary.csv'), index=False)

print(f"\n{'='*75}", flush=True)
print(f"PATH 2 v2.0 COMPLETE — {len(results)} lanes | {time.time()-t0:.0f}s", flush=True)
print(f"Median VAL SARIMAX MAPE : {summary['val_sar_mape'].median():.1f}%", flush=True)
print(f"Median VAL Stacked MAPE : {summary['val_stk_mape'].median():.1f}%", flush=True)
print(f"\n{summary.to_string(index=False)}", flush=True)

# ── Train final models on full stable data ────────────────────────────────────
print(f"\nTraining final models on full stable data...", flush=True)
final_models = {}
for lane in DEMO_LANES:
    y = panel[lane].dropna()
    y = y[y.index >= STABLE_START]
    if len(y) < MIN_TRAIN:
        continue
    exog_cols = get_lane_exog_cols(lane)
    exog_cols = [c for c in exog_cols if c in exog_all.columns]
    exog_lane = exog_all.reindex(y.index)[exog_cols].ffill().bfill()
    xgb_feat  = make_xgb_features(y, exog_lane, exog_cols)
    y_log     = np.log1p(y)
    try:
        m = SARIMAX(y_log, exog=exog_lane,
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
            'last_y': y, 'last_exog': exog_lane,
            'xgb_feat_cols': list(feat_f.columns),
            'feature_importance': fi,
            'reliability': results.get(lane, {}).get('reliability', 'Unknown'),
            'cv': results.get(lane, {}).get('cv', np.nan),
            'path': 'B+C_v2',
            'fx_used': LANE_FX.get(lane),
            'exog_cols': exog_cols,
        }
        print(f"  {lane}: trained (FX={LANE_FX.get(lane,'none')})", flush=True)
    except Exception as e:
        print(f"  {lane}: FAILED — {e}", flush=True)

with open(os.path.join(MDL, 'path2v2_final_models.pkl'), 'wb') as f:
    pickle.dump(final_models, f)
print(f"\nSaved {len(final_models)} Path 2 v2.0 final models → path2v2_final_models.pkl", flush=True)
print("PATH 2 v2.0 DONE.", flush=True)
