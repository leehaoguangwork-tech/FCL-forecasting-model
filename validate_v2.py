"""
Validate v2.0 models (Path 1 and Path 2) on Jan 2025–Mar 2026 out-of-sample period.
Uses the saved final model bundles to avoid re-training.
"""
import os, pickle, warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
MDL  = os.path.join(BASE, 'models', 'antarctica')
OUT  = os.path.join(BASE, 'outputs', 'antarctica')

XGB_CAP = 0.30

def mape(actual, predicted):
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p) & (a > 0)
    if mask.sum() < 2:
        return np.nan
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)

def add_shock_dummies(idx):
    d = pd.DataFrame(index=idx)
    d['shock_covid_spike'] = ((idx >= '2021-06-01') & (idx <= '2022-06-30')).astype(float)
    d['shock_post_crash']  = ((idx >= '2022-07-01') & (idx <= '2023-03-31')).astype(float)
    return d

# Load data
panel    = pd.read_parquet(os.path.join(DATA, 'antarctica_monthly_panel.parquet'))
exog_v1  = pd.read_parquet(os.path.join(DATA, 'exog_features_antarctica.parquet')).rename(columns={'usdcny': 'usd_cny'})
new_exog = pd.read_parquet(os.path.join(DATA, 'new_exog_v2.parquet'))
exog_all = exog_v1.copy()
for col in new_exog.columns:
    exog_all[col] = new_exog[col].reindex(exog_all.index)
exog_all['maersk_proxy'] = exog_all['maersk_proxy'] / 1000.0
exog_all = exog_all.ffill().bfill()

def make_xgb_features(y_series, exog_df, shock_df, exog_cols):
    df = pd.DataFrame(index=y_series.index)
    for lag in [1, 2, 3, 6, 12]:
        df[f'rate_lag{lag}'] = y_series.shift(lag)
    df['rate_roll3'] = y_series.shift(1).rolling(3).mean()
    df['rate_roll6'] = y_series.shift(1).rolling(6).mean()
    df['month_sin']  = np.sin(2 * np.pi * y_series.index.month / 12)
    df['month_cos']  = np.cos(2 * np.pi * y_series.index.month / 12)
    for col in exog_cols:
        if col in exog_df.columns:
            # Use index-aligned reindex to avoid length mismatch
            df[col]         = exog_df[col].reindex(df.index)
            df[f'{col}_l1'] = exog_df[col].reindex(df.index).shift(1)
    for col in shock_df.columns:
        df[col] = shock_df[col].reindex(df.index)
    return df

summary_rows = []

for model_tag, bundle_file in [('path1v2', 'path1v2_final_models.pkl'),
                                ('path2v2', 'path2v2_final_models.pkl')]:
    bundle_fp = os.path.join(MDL, bundle_file)
    if not os.path.exists(bundle_fp):
        print(f"  Skipping {bundle_file} — not found")
        continue

    with open(bundle_fp, 'rb') as f:
        bundle = pickle.load(f)

    print(f"\n{'='*65}")
    print(f"  Validating {model_tag} ({len(bundle)} lanes)")
    print(f"{'='*65}")

    for lane, mdl in bundle.items():
        y = panel[lane].dropna()
        y_train = y[y.index < '2025-01-01']
        y_val   = y[y.index >= '2025-01-01']

        if len(y_val) == 0:
            print(f"  {lane}: no validation data")
            continue

        r_final   = mdl['sarimax_result']
        xgb_final = mdl['xgb_model']
        exog_cols = mdl.get('exog_cols', [])
        xgb_feat_cols = mdl['xgb_feat_cols']
        fx_used = mdl.get('fx_used', 'none')
        has_shock = any('shock_' in c for c in xgb_feat_cols)

        val_actuals, val_sar, val_stk, val_dates = [], [], [], []
        y_rolling = y_train.copy()

        for vt in range(len(y_val)):
            try:
                y_r_log = np.log1p(y_rolling)
                exog_r  = exog_all.reindex(y_rolling.index)[exog_cols].ffill().bfill()

                if has_shock:
                    shock_r = add_shock_dummies(y_rolling.index)
                    exog_r_full = pd.concat([exog_r, shock_r], axis=1)
                else:
                    shock_r = pd.DataFrame(index=y_rolling.index)
                    exog_r_full = exog_r

                r_v = r_final.apply(y_r_log, exog=exog_r_full, refit=False)

                val_idx  = y_val.index[vt]
                exog_vt  = exog_all.reindex([val_idx])[exog_cols].ffill().bfill()

                if has_shock:
                    shock_vt = add_shock_dummies(pd.DatetimeIndex([val_idx]))
                    exog_vt_full = pd.concat([exog_vt, shock_vt], axis=1)
                else:
                    shock_vt = pd.DataFrame(index=pd.DatetimeIndex([val_idx]))
                    exog_vt_full = exog_vt

                fc_log = float(r_v.forecast(steps=1, exog=exog_vt_full).iloc[0])
                sp_v   = float(np.expm1(fc_log))

                # Build XGBoost feature row
                y_ext    = pd.concat([y_rolling, y_val.iloc[:vt+1]])
                exog_ext = pd.concat([exog_r_full, exog_vt_full])
                shock_ext = pd.concat([shock_r, shock_vt])
                xf = make_xgb_features(y_ext, exog_ext, shock_ext, exog_cols).iloc[[-1]].copy()
                xf['sar_pred'] = sp_v
                # Align to trained feature columns
                for c in xgb_feat_cols:
                    if c not in xf.columns:
                        xf[c] = 0.0
                xf = xf[xgb_feat_cols].fillna(0)

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
                print(f"    {lane} vt={vt}: {e}")
                continue

        if len(val_actuals) >= 3:
            m_sar = mape(val_actuals, val_sar)
            m_stk = mape(val_actuals, val_stk)
            vl_df = pd.DataFrame({
                'actual': val_actuals,
                'sarimax_pred': val_sar,
                'stacked_pred': val_stk
            }, index=val_dates)
            vl_df.to_csv(os.path.join(OUT, f'{model_tag}_validation_{lane}.csv'))
            summary_rows.append({
                'model': model_tag, 'lane': lane,
                'fx_used': fx_used or 'none',
                'val_sar_mape': round(m_sar, 2),
                'val_stk_mape': round(m_stk, 2),
                'n_val': len(val_actuals),
            })
            print(f"  {lane:6s} FX={str(fx_used or 'none'):8s}  VAL-SAR {m_sar:6.2f}%  VAL-STK {m_stk:6.2f}%  n={len(val_actuals)}")
        else:
            print(f"  {lane}: insufficient validation predictions ({len(val_actuals)})")

# Save summary
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(os.path.join(OUT, 'v2_validation_summary.csv'), index=False)
print(f"\nSaved: v2_validation_summary.csv")
print(f"\n{df_sum.to_string(index=False)}")
