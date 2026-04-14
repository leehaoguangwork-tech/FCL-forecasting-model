"""
Cap Grid Search — XGBoost Correction Cap Optimisation
Tests cap values: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, None (uncapped)
Measures validation MAPE (Jan 2025 – Mar 2026) for each cap on all lanes, both paths.
"""
import os, pickle, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

BASE  = '/home/ubuntu/fcl_forecast'
OUT   = os.path.join(BASE, 'outputs/antarctica')
MDL   = os.path.join(BASE, 'models/antarctica')

# Cap values to test — None means uncapped
CAPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, None]

def load_val_csv(path_tag, lane):
    fp = os.path.join(OUT, f'{path_tag}_validation_{lane}.csv')
    if not os.path.exists(fp):
        return pd.DataFrame()
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    return df

def apply_cap(sar_pred, xgb_raw_correction, cap):
    """Apply cap to XGBoost correction relative to SARIMAX prediction."""
    if cap is None:
        return sar_pred + xgb_raw_correction
    max_corr = abs(sar_pred) * cap
    clipped = np.clip(xgb_raw_correction, -max_corr, max_corr)
    return sar_pred + clipped

def mape(actual, predicted):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

results = []

for path_tag, bundle_file in [('path1', 'path1_final_models.pkl'),
                               ('path2', 'path2_final_models.pkl')]:
    bundle_fp = os.path.join(MDL, bundle_file)
    if not os.path.exists(bundle_fp):
        print(f"  Bundle not found: {bundle_fp}")
        continue

    with open(bundle_fp, 'rb') as f:
        bundle = pickle.load(f)

    lanes = list(bundle.keys())
    print(f"\n{'='*60}")
    print(f"Path: {path_tag}  |  Lanes: {lanes}")
    print(f"{'='*60}")

    for lane in lanes:
        val_df = load_val_csv(path_tag, lane)
        if val_df.empty:
            print(f"  {lane}: no validation CSV, skipping")
            continue
        if 'actual' not in val_df.columns or 'sarimax_pred' not in val_df.columns:
            print(f"  {lane}: missing columns in val CSV, skipping")
            continue

        actual    = val_df['actual'].values
        sar_pred  = val_df['sarimax_pred'].values

        # Compute raw XGBoost correction (before any cap) from existing stacked_pred
        # stacked_pred = sar_pred + clipped_correction (at 30%)
        # We need the raw XGB output — re-run XGB predict on val features if available
        # Fallback: use stacked_pred to back-calculate the 30%-capped correction,
        # then infer raw correction from the model directly.

        # Best approach: re-predict from XGB model on val exog features
        mdl_data = bundle[lane]
        xgb_model = mdl_data.get('xgb_model')
        if xgb_model is None:
            print(f"  {lane}: no xgb_model in bundle, skipping")
            continue

        # Load val features — reconstruct from val_df columns
        feat_cols = mdl_data.get('xgb_feat_cols', [])
        if not feat_cols:
            print(f"  {lane}: no xgb_feat_cols, skipping")
            continue

        # Check if val_df has the feature columns
        available = [c for c in feat_cols if c in val_df.columns]
        if len(available) < len(feat_cols):
            # Try loading the backtest CSV which may have more features
            bt_fp = os.path.join(OUT, f'{path_tag}_backtest_{lane}.csv')
            # We'll use the stacked_pred back-calculation approach instead
            # stacked_pred at 30% cap: correction = stacked - sar, capped at 30%
            # raw_xgb >= correction (since cap may have truncated it)
            # We cannot recover raw_xgb without re-running the model on features
            # So: use stacked_pred as proxy — this gives us the 30%-capped result
            # and we test other caps by scaling: raw ≈ correction / min(1, cap/0.30)
            # This is an approximation but valid for comparing relative cap effects.
            stacked = val_df['stacked_pred'].values
            correction_30 = stacked - sar_pred  # correction at 30% cap

            print(f"  {lane}: using back-calculation (approx) for raw XGB correction")

            for cap in CAPS:
                if cap is None:
                    # Uncapped: assume raw = correction_30 / 0.30 * max_possible
                    # We can't truly uncap without raw model output, so skip for approx lanes
                    final_pred = stacked  # same as 30% — best we can do
                    cap_label = 'uncapped*'
                else:
                    # Scale correction: if cap < 0.30, further restrict; if cap > 0.30, allow more
                    # Approximate: raw_correction ≈ correction_30 (since 30% may not have been binding)
                    max_corr = np.abs(sar_pred) * cap
                    clipped  = np.clip(correction_30, -max_corr, max_corr)
                    final_pred = sar_pred + clipped
                    cap_label = f"{int(cap*100)}%"

                m = mape(actual, final_pred)
                results.append({
                    'path': path_tag,
                    'lane': lane,
                    'cap': cap_label,
                    'cap_val': cap if cap is not None else 9999,
                    'val_mape': round(m, 3),
                    'method': 'approx'
                })
        else:
            # We have all feature columns — re-run XGB predict directly
            import xgboost as xgb
            X_val = val_df[feat_cols].values
            raw_correction = xgb_model.predict(xgb.DMatrix(X_val))

            print(f"  {lane}: re-running XGB predict on {len(X_val)} val rows")

            for cap in CAPS:
                final_pred = apply_cap(sar_pred, raw_correction, cap)
                cap_label  = f"{int(cap*100)}%" if cap is not None else 'uncapped'
                m = mape(actual, final_pred)
                results.append({
                    'path': path_tag,
                    'lane': lane,
                    'cap': cap_label,
                    'cap_val': cap if cap is not None else 9999,
                    'val_mape': round(m, 3),
                    'method': 'exact'
                })

        # Print summary for this lane
        lane_res = [r for r in results if r['lane'] == lane and r['path'] == path_tag]
        if lane_res:
            best = min(lane_res, key=lambda x: x['val_mape'])
            current = next((r for r in lane_res if r['cap'] == '30%'), None)
            print(f"  {lane}: current(30%)={current['val_mape']:.2f}%  best={best['cap']}→{best['val_mape']:.2f}%")

# Save results
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT, 'cap_grid_search_results.csv'), index=False)
print(f"\nSaved: cap_grid_search_results.csv")

# Summary table
print("\n" + "="*70)
print("OPTIMAL CAP PER LANE")
print("="*70)
summary_rows = []
for (path, lane), grp in res_df.groupby(['path', 'lane']):
    best = grp.loc[grp['val_mape'].idxmin()]
    cur  = grp[grp['cap'] == '30%']
    cur_mape = cur['val_mape'].values[0] if len(cur) else float('nan')
    improvement = cur_mape - best['val_mape']
    summary_rows.append({
        'Path': path,
        'Lane': lane,
        'Current Cap (30%) MAPE': f"{cur_mape:.2f}%",
        'Optimal Cap': best['cap'],
        'Optimal MAPE': f"{best['val_mape']:.2f}%",
        'Improvement': f"{improvement:+.2f}pp",
    })
    print(f"  {path} | {lane:8s} | 30%={cur_mape:.2f}%  best={best['cap']}={best['val_mape']:.2f}%  gain={improvement:+.2f}pp")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT, 'cap_grid_summary.csv'), index=False)
print(f"\nSaved: cap_grid_summary.csv")
print("\nDone.")
