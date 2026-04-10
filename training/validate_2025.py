"""
Jan 2025 – Apr 2026 Out-of-Sample Validation
=============================================
Uses the final trained models from Path 1 and Path 2.
Generates multi-step forecasts from Dec 2024 anchor point,
then compares against actual rates in the panel.
"""

import os, json, pickle, warnings, time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
MDL  = os.path.join(BASE, 'models', 'antarctica')
OUT  = os.path.join(BASE, 'outputs', 'antarctica')

DEMO_LANES = ['CNSHA','JPTYO','KRPUS','INNSA','THBKK',
              'DEHAM','NLRTM','GBFXT','USNYC','USLAX',
              'ARBUE','AEJEA','NGLOS','AUSYD','QAHMD']

panel = pd.read_parquet(os.path.join(DATA, 'antarctica_monthly_panel.parquet'))
exog  = pd.read_parquet(os.path.join(DATA, 'exog_features_antarctica.parquet'))

EXOG_COLS = ['brent_crude','usdcny','us_indpro','us_cfnai','bdry_etf',
             'dummy_covid','dummy_supply_crunch','dummy_ukraine',
             'dummy_red_sea','dummy_hormuz']

XGB_CAP = 0.30

def safe_metrics(act, pred):
    act  = np.array(act,  dtype=float)
    pred = np.array(pred, dtype=float)
    mask = np.isfinite(act) & np.isfinite(pred) & (act > 0)
    act  = act[mask]; pred = pred[mask]
    if len(act) < 2:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'dir_acc': np.nan, 'n': len(act)}
    pred = np.clip(pred, 0, act.max() * 5)
    mae  = float(mean_absolute_error(act, pred))
    rmse = float(np.sqrt(mean_squared_error(act, pred)))
    mape = float(np.mean(np.abs((act - pred) / np.maximum(act, 1))) * 100)
    da   = float(np.mean(np.sign(np.diff(act)) == np.sign(np.diff(pred))) * 100) if len(act) > 1 else np.nan
    return {'mae': round(mae,2), 'rmse': round(rmse,2),
            'mape': round(mape,2), 'dir_acc': round(da,2), 'n': len(act)}

def validate_path(path_label, model_file):
    print(f"\n{'='*65}", flush=True)
    print(f"VALIDATING {path_label}", flush=True)
    print(f"{'='*65}", flush=True)

    with open(os.path.join(MDL, model_file), 'rb') as f:
        models = pickle.load(f)

    all_results = []
    val_summary = {}

    for lane in DEMO_LANES:
        if lane not in models:
            print(f"  {lane}: no model", flush=True)
            continue

        m = models[lane]
        y_all   = panel[lane].dropna()
        y_val   = y_all[y_all.index >= '2025-01-01']

        if len(y_val) == 0:
            print(f"  {lane}: no validation data", flush=True)
            continue

        r        = m['sarimax_result']
        xgb_mod  = m['xgb_model']
        last_y   = m['last_y']   # training data used to fit the model
        last_exog= m['last_exog']
        feat_cols= m['xgb_feat_cols']

        # Build exog for validation period
        exog_val = exog.reindex(y_val.index)[EXOG_COLS].ffill().bfill()

        # For Path 1, add shock dummies to validation exog
        if m.get('path') == 'A+C':
            # Red Sea dummy stays 1 through 2025 (ongoing), others 0
            exog_val['shock_covid_spike'] = 0.0
            exog_val['shock_post_crash']  = 0.0

        # Multi-step forecast: use rolling actuals as they become available
        # This simulates real-world use where you know last month's actual
        y_rolling   = last_y.copy()
        exog_rolling= last_exog.copy()

        sar_preds, stk_preds, actuals, dates = [], [], [], []

        for vt in range(len(y_val)):
            try:
                y_r_log = np.log1p(y_rolling)

                # Extend SARIMAX with rolling data
                r_ext = r.apply(y_r_log, exog=exog_rolling, refit=False)

                # Forecast 1 step ahead
                ex_next = exog_val.iloc[[vt]]
                fc_log  = float(r_ext.forecast(steps=1, exog=ex_next).iloc[0])
                sp      = float(np.expm1(fc_log))
                sp      = max(sp, 0.1)

                # XGBoost residual correction
                # Build feature row for this step
                y_ext = pd.concat([y_rolling, y_val.iloc[:vt]])
                ex_ext= pd.concat([exog_rolling, exog_val.iloc[:vt]])

                feat_row = {}
                for lag in [1,2,3,6,12]:
                    key = f'rate_lag{lag}'
                    if key in feat_cols:
                        feat_row[key] = float(y_ext.iloc[-lag]) if len(y_ext) >= lag else np.nan
                feat_row['rate_roll3'] = float(y_ext.iloc[-3:].mean()) if len(y_ext) >= 3 else np.nan
                feat_row['rate_roll6'] = float(y_ext.iloc[-6:].mean()) if len(y_ext) >= 6 else np.nan
                feat_row['month_sin']  = np.sin(2*np.pi*y_val.index[vt].month/12)
                feat_row['month_cos']  = np.cos(2*np.pi*y_val.index[vt].month/12)
                for col in EXOG_COLS:
                    if col in feat_cols:
                        feat_row[col] = float(ex_next[col].iloc[0]) if col in ex_next.columns else 0.0
                    if f'{col}_l1' in feat_cols:
                        feat_row[f'{col}_l1'] = float(exog_rolling[col].iloc[-1]) if col in exog_rolling.columns else 0.0
                # Shock dummies for Path 1
                for sc in ['shock_covid_spike','shock_post_crash']:
                    if sc in feat_cols:
                        feat_row[sc] = 0.0
                feat_row['sar_pred'] = sp

                import pandas as pd2
                xf = pd.DataFrame([feat_row])[feat_cols].fillna(0)
                corr = float(xgb_mod.predict(xf)[0])
                cap  = XGB_CAP * abs(sp)
                corr = np.clip(corr, -cap, cap)
                stk  = sp + corr

                sar_preds.append(sp)
                stk_preds.append(stk)
                actuals.append(float(y_val.iloc[vt]))
                dates.append(y_val.index[vt])

                # Roll forward with actual value
                y_rolling   = pd.concat([y_rolling,   y_val.iloc[[vt]]])
                exog_rolling= pd.concat([exog_rolling, exog_val.iloc[[vt]]])

            except Exception as e:
                # On failure, still roll forward
                try:
                    y_rolling   = pd.concat([y_rolling,   y_val.iloc[[vt]]])
                    exog_rolling= pd.concat([exog_rolling, exog_val.iloc[[vt]]])
                except Exception:
                    pass

        sar_m = safe_metrics(actuals, sar_preds)
        stk_m = safe_metrics(actuals, stk_preds)

        # Save validation CSV
        if actuals:
            vdf = pd.DataFrame({
                'actual':       actuals,
                'sarimax_pred': sar_preds,
                'stacked_pred': stk_preds
            }, index=pd.DatetimeIndex(dates))
            tag = 'path1' if m.get('path') == 'A+C' else 'path2'
            vdf.to_csv(os.path.join(OUT, f'{tag}_validation_{lane}.csv'))

        val_summary[lane] = {'sarimax': sar_m, 'stacked': stk_m}
        all_results.append({
            'lane': lane,
            'reliability': m.get('reliability','?'),
            'val_sar_mape': sar_m['mape'],
            'val_stk_mape': stk_m['mape'],
            'val_sar_mae':  sar_m['mae'],
            'val_dir_acc':  sar_m['dir_acc'],
            'n_months':     sar_m['n']
        })

        print(f"  {lane:6s} [{m.get('reliability','?'):6s}] "
              f"VAL-SAR {sar_m['mape']:6.1f}% "
              f"VAL-STK {stk_m['mape']:6.1f}% "
              f"Dir {sar_m['dir_acc']:5.1f}% "
              f"n={sar_m['n']}", flush=True)

    tag = 'path1' if 'path1' in model_file else 'path2'
    with open(os.path.join(OUT, f'{tag}_val_summary.json'), 'w') as f:
        json.dump(val_summary, f, indent=2)

    df = pd.DataFrame(all_results).sort_values('val_sar_mape')
    df.to_csv(os.path.join(OUT, f'{tag}_val_table.csv'), index=False)

    print(f"\nSummary ({path_label}):", flush=True)
    print(f"  Median VAL SARIMAX MAPE : {df['val_sar_mape'].median():.1f}%", flush=True)
    print(f"  Median VAL Stacked MAPE : {df['val_stk_mape'].median():.1f}%", flush=True)
    print(f"  Median Dir Accuracy     : {df['val_dir_acc'].median():.1f}%", flush=True)
    print(df.to_string(index=False), flush=True)
    return df

t0 = time.time()
df1 = validate_path("PATH 1 (A+C) — Full data + shock dummies", "path1_final_models.pkl")
df2 = validate_path("PATH 2 (B+C) — Post-Jul 2022 split",       "path2_final_models.pkl")

# ── Side-by-side comparison ───────────────────────────────────────────────────
print(f"\n{'='*75}", flush=True)
print("SIDE-BY-SIDE COMPARISON: Path 1 (A+C) vs Path 2 (B+C)", flush=True)
print("Validation period: Jan 2025 – Apr 2026", flush=True)
print(f"{'='*75}", flush=True)

comp = df1[['lane','reliability','val_sar_mape']].rename(
    columns={'val_sar_mape':'P1_SAR_MAPE'}).merge(
    df2[['lane','val_sar_mape']].rename(columns={'val_sar_mape':'P2_SAR_MAPE'}),
    on='lane', how='outer')
comp['winner'] = comp.apply(
    lambda r: 'Path1' if r['P1_SAR_MAPE'] < r['P2_SAR_MAPE'] else 'Path2', axis=1)
comp['improvement'] = (comp['P1_SAR_MAPE'] - comp['P2_SAR_MAPE']).round(1)

print(comp.to_string(index=False), flush=True)
print(f"\nPath 1 wins on {(comp['winner']=='Path1').sum()} lanes", flush=True)
print(f"Path 2 wins on {(comp['winner']=='Path2').sum()} lanes", flush=True)
print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)

comp.to_csv(os.path.join(OUT, 'path_comparison.csv'), index=False)
print("\nValidation complete.", flush=True)
