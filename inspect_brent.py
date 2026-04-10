"""
Inspect how Brent crude oil enters the model:
- SARIMAX: linear coefficient (params Series indexed by name)
- XGBoost: feature importance (non-linear tree splits)
"""
import pickle, pandas as pd, numpy as np

def get_sarimax_brent(res):
    """Extract brent_crude coefficient and p-value from a SARIMAX result."""
    params = res.params
    pvals  = res.pvalues
    # params may be a Series with string index (exog names) or numeric
    if hasattr(params, 'index') and 'brent_crude' in params.index:
        coef = float(params['brent_crude'])
        pval = float(pvals['brent_crude'])
        return coef, pval
    # fallback: use exog_names list
    exog_names = res.model.exog_names
    if 'brent_crude' in exog_names:
        idx = exog_names.index('brent_crude')
        coef = float(params.iloc[idx])
        pval = float(pvals.iloc[idx])
        return coef, pval
    return None, None

# ── PATH 1 ────────────────────────────────────────────────────────────────────
with open('models/antarctica/path1_final_models.pkl','rb') as f:
    p1 = pickle.load(f)

print("=" * 72)
print("PATH 1 — SARIMAX  (pure LINEAR coefficient on brent_crude)")
print("  Interpretation: for every +1 USD/bbl in Brent, forecast changes by coef ATD")
print("=" * 72)
print(f"  {'Lane':<8}  {'Coef (ATD/bbl)':>16}  {'p-value':>10}  Sig")
print(f"  {'-'*8}  {'-'*16}  {'-'*10}  ---")
for lane, m in p1.items():
    coef, pval = get_sarimax_brent(m['sarimax_result'])
    if coef is not None:
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {lane:<8}  {coef:>+16.6f}  {pval:>10.4f}  {sig}")

print()
print("=" * 72)
print("PATH 1 — XGBoost  (NON-LINEAR — tree-based, captures threshold effects)")
print("  Feature importance = fraction of splits using that feature")
print("=" * 72)
print(f"  {'Lane':<8}  {'brent_crude':>12}  {'brent_crude_l1':>15}  {'combined':>10}  {'rank (of all feats)':>20}")
print(f"  {'-'*8}  {'-'*12}  {'-'*15}  {'-'*10}  {'-'*20}")
for lane, m in p1.items():
    xgb = m['xgb_model']
    feat_cols = m['xgb_feat_cols']
    imp = dict(zip(feat_cols, xgb.feature_importances_))
    b0 = imp.get('brent_crude', 0)
    b1 = imp.get('brent_crude_l1', 0)
    combined = b0 + b1
    # rank combined vs all features
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
    rank_b0 = next((i+1 for i,(k,v) in enumerate(sorted_imp) if k=='brent_crude'), None)
    rank_b1 = next((i+1 for i,(k,v) in enumerate(sorted_imp) if k=='brent_crude_l1'), None)
    print(f"  {lane:<8}  {b0:>12.4f}  {b1:>15.4f}  {combined:>10.4f}  rank: {rank_b0} / {rank_b1} (curr/lag)")

# ── PATH 2 ────────────────────────────────────────────────────────────────────
with open('models/antarctica/path2_final_models.pkl','rb') as f:
    p2 = pickle.load(f)

print()
print("=" * 72)
print("PATH 2 — SARIMAX  (pure LINEAR coefficient on brent_crude)")
print("=" * 72)
print(f"  {'Lane':<8}  {'Coef (ATD/bbl)':>16}  {'p-value':>10}  Sig")
print(f"  {'-'*8}  {'-'*16}  {'-'*10}  ---")
for lane, m in p2.items():
    coef, pval = get_sarimax_brent(m['sarimax_result'])
    if coef is not None:
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {lane:<8}  {coef:>+16.6f}  {pval:>10.4f}  {sig}")

print()
print("=" * 72)
print("PATH 2 — XGBoost  (NON-LINEAR)")
print("=" * 72)
print(f"  {'Lane':<8}  {'brent_crude':>12}  {'brent_crude_l1':>15}  {'combined':>10}")
print(f"  {'-'*8}  {'-'*12}  {'-'*15}  {'-'*10}")
for lane, m in p2.items():
    xgb = m['xgb_model']
    feat_cols = m['xgb_feat_cols']
    imp = dict(zip(feat_cols, xgb.feature_importances_))
    b0 = imp.get('brent_crude', 0)
    b1 = imp.get('brent_crude_l1', 0)
    print(f"  {lane:<8}  {b0:>12.4f}  {b1:>15.4f}  {b0+b1:>10.4f}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("SUMMARY — Nature of Brent crude relationship in the model")
print("=" * 72)
print("""
  Component   | Relationship  | How it works
  ------------|---------------|----------------------------------------------
  SARIMAX     | LINEAR        | Fixed coefficient β: +1 USD/bbl → +β ATD
              |               | Coefficient is constant across all price levels
  XGBoost     | NON-LINEAR    | Decision tree splits: captures threshold effects,
              |               | diminishing returns, and interaction with other
              |               | features (e.g. Brent × Red Sea dummy)
  ------------|---------------|----------------------------------------------
  COMBINED    | HYBRID        | Linear base trend (SARIMAX) + non-linear
              |               | residual correction (XGBoost, capped at ±30%)
""")
