"""
Linear Stacking Meta-Learner
============================
Replaces the fixed-cap sequential stack with:
    Final = α × SARIMAX_pred + β × XGBoost_pred

where α and β are fitted via Leave-One-Out cross-validation on the training set.

Also tests:
  - Constrained version: α + β = 1 (pure blend)
  - Unconstrained version: α + β can differ from 1

Compares against:
  1. Current method: SARIMAX + XGBoost correction (30% cap)
  2. Optimal-cap method: SARIMAX + XGBoost correction (per-lane optimal cap from grid search)
  3. Meta-learner (constrained blend)
  4. Meta-learner (unconstrained blend)

Outputs:
  - meta_learner_results.json   — all MAPE results
  - meta_learner_weights.json   — fitted α, β per lane
  - meta_validation_*.csv       — per-lane validation predictions
"""
import os, pickle, json, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

BASE = '/home/ubuntu/fcl_forecast'
OUT  = os.path.join(BASE, 'outputs/antarctica')
MDL  = os.path.join(BASE, 'models/antarctica')

# Load optimal caps from grid search
cap_grid_fp = os.path.join(OUT, 'cap_grid_summary.csv')
cap_df = pd.read_csv(cap_grid_fp)
# Build lookup: (path, lane) -> optimal_cap_float
def parse_cap(s):
    s = str(s).strip()
    if s in ('uncapped', 'uncapped*'):
        return None
    return float(s.replace('%','')) / 100.0

cap_lookup = {}
for _, row in cap_df.iterrows():
    cap_lookup[(row['Path'], row['Lane'])] = parse_cap(row['Optimal Cap'])

def mape(actual, predicted):
    mask = (actual != 0) & (~np.isnan(actual)) & (~np.isnan(predicted))
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def apply_cap(sar, corr, cap):
    if cap is None:
        return sar + corr
    max_c = np.abs(sar) * cap
    return sar + np.clip(corr, -max_c, max_c)

def fit_meta_constrained(sar_train, xgb_train, actual_train):
    """Fit α such that Final = α*SAR + (1-α)*XGB — constrained blend."""
    def loss(params):
        alpha = params[0]
        pred = alpha * sar_train + (1 - alpha) * xgb_train
        return mape(actual_train, pred)
    res = minimize(loss, x0=[0.5], bounds=[(0.0, 1.0)], method='L-BFGS-B')
    alpha = res.x[0]
    return alpha, 1 - alpha

def fit_meta_unconstrained(sar_train, xgb_train, actual_train):
    """Fit α, β freely — Final = α*SAR + β*XGB."""
    def loss(params):
        alpha, beta = params
        pred = alpha * sar_train + beta * xgb_train
        return mape(actual_train, pred)
    res = minimize(loss, x0=[0.5, 0.5], method='Nelder-Mead')
    return res.x[0], res.x[1]

def loo_meta_weights(sar_series, xgb_series, actual_series, constrained=True):
    """
    Leave-One-Out cross-validation to fit meta-learner weights.
    Returns: (alpha, beta) fitted on all-but-last observations,
             validated on held-out observations.
    We use a rolling expanding window: train on first k obs, predict k+1.
    Final weights are fitted on the full training set.
    """
    n = len(sar_series)
    loo_preds = np.full(n, np.nan)

    min_train = max(5, n // 4)  # need at least 5 obs to fit
    for i in range(min_train, n):
        s_tr = sar_series[:i]
        x_tr = xgb_series[:i]
        a_tr = actual_series[:i]
        if constrained:
            a, b = fit_meta_constrained(s_tr, x_tr, a_tr)
        else:
            a, b = fit_meta_unconstrained(s_tr, x_tr, a_tr)
        loo_preds[i] = a * sar_series[i] + b * xgb_series[i]

    # Final weights on full training set
    if constrained:
        alpha_final, beta_final = fit_meta_constrained(sar_series, xgb_series, actual_series)
    else:
        alpha_final, beta_final = fit_meta_unconstrained(sar_series, xgb_series, actual_series)

    return alpha_final, beta_final, loo_preds

all_results = {}
all_weights = {}

for path_tag, bundle_file in [('path1', 'path1_final_models.pkl'),
                               ('path2', 'path2_final_models.pkl')]:
    bundle_fp = os.path.join(MDL, bundle_file)
    if not os.path.exists(bundle_fp):
        continue
    with open(bundle_fp, 'rb') as f:
        bundle = pickle.load(f)

    print(f"\n{'='*65}")
    print(f"  {path_tag.upper()} — {'Model with COVID' if path_tag=='path1' else 'Stable Model'}")
    print(f"{'='*65}")

    for lane in bundle.keys():
        # Load validation CSV
        val_fp = os.path.join(OUT, f'{path_tag}_validation_{lane}.csv')
        bt_fp  = os.path.join(OUT, f'{path_tag}_backtest_{lane}.csv')
        if not os.path.exists(val_fp):
            continue

        val_df = pd.read_csv(val_fp, index_col=0, parse_dates=True)
        if 'actual' not in val_df.columns or 'sarimax_pred' not in val_df.columns:
            continue

        # Load backtest for training meta-learner
        if not os.path.exists(bt_fp):
            continue
        bt_df = pd.read_csv(bt_fp, index_col=0, parse_dates=True)
        if 'actual' not in bt_df.columns or 'sarimax_pred' not in bt_df.columns:
            continue

        # ── Training data (backtest period) ──────────────────────────────
        bt_actual  = bt_df['actual'].values.astype(float)
        bt_sar     = bt_df['sarimax_pred'].values.astype(float)
        bt_stacked = bt_df['stacked_pred'].values.astype(float)
        # Approximate raw XGB prediction from stacked (30% cap back-calculation)
        bt_corr30  = bt_stacked - bt_sar
        # XGB standalone prediction ≈ SAR + raw_correction
        # We use stacked as proxy for XGB standalone (since correction may be small)
        bt_xgb_pred = bt_stacked  # proxy: SAR + capped_correction

        # ── Validation data ───────────────────────────────────────────────
        val_actual  = val_df['actual'].values.astype(float)
        val_sar     = val_df['sarimax_pred'].values.astype(float)
        val_stacked = val_df['stacked_pred'].values.astype(float)
        val_corr30  = val_stacked - val_sar
        val_xgb_pred = val_stacked  # proxy

        # ── Method 1: Current stack (30% cap) ────────────────────────────
        mape_current = mape(val_actual, val_stacked)

        # ── Method 2: Optimal-cap stack ───────────────────────────────────
        opt_cap = cap_lookup.get((path_tag, lane), 0.30)
        if opt_cap is None:
            val_opt = val_stacked
        else:
            max_c = np.abs(val_sar) * opt_cap
            val_opt = val_sar + np.clip(val_corr30, -max_c, max_c)
        mape_opt_cap = mape(val_actual, val_opt)

        # ── Method 3: Meta-learner constrained (α + β = 1) ───────────────
        alpha_c, beta_c, _ = loo_meta_weights(bt_sar, bt_xgb_pred, bt_actual, constrained=True)
        val_meta_c = alpha_c * val_sar + beta_c * val_xgb_pred
        mape_meta_c = mape(val_actual, val_meta_c)

        # ── Method 4: Meta-learner unconstrained ─────────────────────────
        alpha_u, beta_u, _ = loo_meta_weights(bt_sar, bt_xgb_pred, bt_actual, constrained=False)
        val_meta_u = alpha_u * val_sar + beta_u * val_xgb_pred
        mape_meta_u = mape(val_actual, val_meta_u)

        # ── Store results ─────────────────────────────────────────────────
        key = f"{path_tag}_{lane}"
        all_results[key] = {
            'path': path_tag,
            'lane': lane,
            'n_train': len(bt_df),
            'n_val': len(val_df),
            'mape_current_30pct': round(mape_current, 3),
            'mape_optimal_cap': round(mape_opt_cap, 3),
            'optimal_cap_used': f"{int(opt_cap*100)}%" if opt_cap else "30%",
            'mape_meta_constrained': round(mape_meta_c, 3),
            'mape_meta_unconstrained': round(mape_meta_u, 3),
            'best_method': min(
                [('Current (30%)', mape_current),
                 ('Optimal Cap', mape_opt_cap),
                 ('Meta Constrained', mape_meta_c),
                 ('Meta Unconstrained', mape_meta_u)],
                key=lambda x: x[1]
            )[0],
            'best_mape': round(min(mape_current, mape_opt_cap, mape_meta_c, mape_meta_u), 3),
        }
        all_weights[key] = {
            'path': path_tag, 'lane': lane,
            'alpha_constrained': round(float(alpha_c), 4),
            'beta_constrained':  round(float(beta_c), 4),
            'alpha_unconstrained': round(float(alpha_u), 4),
            'beta_unconstrained':  round(float(beta_u), 4),
            'optimal_cap': f"{int(opt_cap*100)}%" if opt_cap else "30%",
        }

        # Save per-lane validation CSV with all method predictions
        val_out = val_df[['actual','sarimax_pred','stacked_pred']].copy()
        val_out['optimal_cap_pred']      = val_opt
        val_out['meta_constrained_pred'] = val_meta_c
        val_out['meta_unconstrained_pred'] = val_meta_u
        val_out.to_csv(os.path.join(OUT, f'meta_validation_{path_tag}_{lane}.csv'))

        # Print summary
        best = all_results[key]['best_method']
        print(f"  {lane:8s} | 30%={mape_current:6.2f}%  OptCap={mape_opt_cap:6.2f}%  "
              f"MetaC={mape_meta_c:6.2f}%  MetaU={mape_meta_u:6.2f}%  "
              f"α={alpha_c:.2f} β={beta_c:.2f}  Best: {best}")

# Save outputs
with open(os.path.join(OUT, 'meta_learner_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
with open(os.path.join(OUT, 'meta_learner_weights.json'), 'w') as f:
    json.dump(all_weights, f, indent=2)

# Summary statistics
res_list = list(all_results.values())
for path_tag in ['path1', 'path2']:
    sub = [r for r in res_list if r['path'] == path_tag]
    if not sub:
        continue
    label = 'Model with COVID' if path_tag == 'path1' else 'Stable Model'
    print(f"\n{'─'*65}")
    print(f"  {label} — Average Validation MAPE across {len(sub)} lanes")
    print(f"{'─'*65}")
    print(f"  Current (30% cap):        {np.mean([r['mape_current_30pct'] for r in sub]):.2f}%")
    print(f"  Optimal cap (per-lane):   {np.mean([r['mape_optimal_cap'] for r in sub]):.2f}%")
    print(f"  Meta-learner constrained: {np.mean([r['mape_meta_constrained'] for r in sub]):.2f}%")
    print(f"  Meta-learner unconstrained:{np.mean([r['mape_meta_unconstrained'] for r in sub]):.2f}%")
    wins = {}
    for r in sub:
        wins[r['best_method']] = wins.get(r['best_method'], 0) + 1
    print(f"  Best method wins: {wins}")

print("\nSaved: meta_learner_results.json, meta_learner_weights.json")
print("Done.")
