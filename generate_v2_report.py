"""
Generate the FCL Forecasting v2.0 Upgrade Report:
  - Model architecture and rationale
  - Path 1 v1.0 vs v2.0 comparison (lane-specific FX + Maersk proxy)
  - Path 2 v1.0 (30% cap) vs v1.0 (10% cap) comparison
  - Why Path 2 v2.0 with new FX features failed
  - Recommendations and next steps
"""
import os, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE = '/home/ubuntu/fcl_forecast'
OUT  = os.path.join(BASE, 'outputs', 'antarctica')
PLOT = os.path.join(BASE, 'outputs', 'v2_report_plots')
os.makedirs(PLOT, exist_ok=True)

LANES = ['CNSHA','JPTYO','KRPUS','INNSA','THBKK','DEHAM','NLRTM','GBFXT',
         'USNYC','USLAX','ARBUE','AEJEA','NGLOS','AUSYD','QAHMD']

LANE_FX = {
    'CNSHA':'USD/CNY','JPTYO':'USD/CNY','KRPUS':'USD/CNY',
    'INNSA':'USD/INR','THBKK':'USD/CNY',
    'DEHAM':'USD/EUR','NLRTM':'USD/EUR','GBFXT':'USD/EUR',
    'USNYC':'None (domestic)','USLAX':'None (domestic)',
    'ARBUE':'USD/ARS','AEJEA':'None (AED pegged)','NGLOS':'USD/NGN',
    'AUSYD':'USD/AUD','QAHMD':'None (AED pegged)',
}

def mape_from_csv(fp, pred_col=None):
    if not os.path.exists(fp): return np.nan
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    act = df['actual'].values
    if pred_col and pred_col in df.columns:
        pred = df[pred_col].values
    elif 'stacked_pred' in df.columns:
        pred = df['stacked_pred'].values
    elif 'stk_pred' in df.columns:
        pred = df['stk_pred'].values
    elif 'cap30_pred' in df.columns:
        pred = df['cap30_pred'].values
    else:
        return np.nan
    mask = np.isfinite(act) & np.isfinite(pred) & (act > 0)
    if mask.sum() < 2: return np.nan
    return float(np.mean(np.abs((act[mask]-pred[mask])/act[mask]))*100)

def sar_mape_from_csv(fp):
    if not os.path.exists(fp): return np.nan
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    act = df['actual'].values
    if 'sarimax_pred' in df.columns:
        pred = df['sarimax_pred'].values
    else:
        return np.nan
    mask = np.isfinite(act) & np.isfinite(pred) & (act > 0)
    if mask.sum() < 2: return np.nan
    return float(np.mean(np.abs((act[mask]-pred[mask])/act[mask]))*100)

# ── Collect comparison data ───────────────────────────────────────────────────
rows = []
for lane in LANES:
    p1v1_stk = mape_from_csv(f'{OUT}/path1_validation_{lane}.csv')
    p1v1_sar = sar_mape_from_csv(f'{OUT}/path1_validation_{lane}.csv')
    p1v2_stk = mape_from_csv(f'{OUT}/path1v2_validation_{lane}.csv')
    p1v2_sar = sar_mape_from_csv(f'{OUT}/path1v2_validation_{lane}.csv')
    p2v1_30  = mape_from_csv(f'{OUT}/path2_validation_{lane}.csv')
    p2v1_10  = mape_from_csv(f'{OUT}/cap10_validation_path2_{lane}.csv', pred_col='cap10_pred')
    rows.append({
        'lane': lane,
        'fx': LANE_FX.get(lane,'—'),
        'p1v1_sar': round(p1v1_sar,2) if np.isfinite(p1v1_sar) else np.nan,
        'p1v1_stk': round(p1v1_stk,2) if np.isfinite(p1v1_stk) else np.nan,
        'p1v2_sar': round(p1v2_sar,2) if np.isfinite(p1v2_sar) else np.nan,
        'p1v2_stk': round(p1v2_stk,2) if np.isfinite(p1v2_stk) else np.nan,
        'p2v1_30':  round(p2v1_30,2)  if np.isfinite(p2v1_30)  else np.nan,
        'p2v1_10':  round(p2v1_10,2)  if np.isfinite(p2v1_10)  else np.nan,
    })

df = pd.DataFrame(rows)
df['p1_delta'] = df['p1v1_stk'] - df['p1v2_stk']  # positive = v2 better
df['p2_delta'] = df['p2v1_30']  - df['p2v1_10']    # positive = 10% cap better
df.to_csv(f'{OUT}/v2_full_comparison.csv', index=False)
print(df.to_string(index=False))

# ── Chart 1: Path 1 v1 vs v2 stacked MAPE ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14,6))
x = np.arange(len(LANES))
w = 0.35
bars1 = ax.bar(x - w/2, df['p1v1_stk'], w, label='Model with COVID v1.0 (USD/CNY)', color='#4C72B0', alpha=0.85)
bars2 = ax.bar(x + w/2, df['p1v2_stk'], w, label='Model with COVID v2.0 (Lane FX + Maersk)', color='#55A868', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(LANES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Validation MAPE (%)', fontsize=11)
ax.set_title('Model with COVID: v1.0 vs v2.0 — Validation MAPE by Lane\n(Lower is better)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(df['p1v1_stk'].median(), color='#4C72B0', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(df['p1v2_stk'].median(), color='#55A868', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylim(0, max(df['p1v1_stk'].max(), df['p1v2_stk'].max()) * 1.15)
for bar in bars1:
    h = bar.get_height()
    if np.isfinite(h): ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=7, color='#4C72B0')
for bar in bars2:
    h = bar.get_height()
    if np.isfinite(h): ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=7, color='#55A868')
plt.tight_layout()
plt.savefig(f'{PLOT}/path1_v1_vs_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved path1_v1_vs_v2.png")

# ── Chart 2: Path 2 30% cap vs 10% cap ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14,6))
bars1 = ax.bar(x - w/2, df['p2v1_30'], w, label='Stable Model — 30% cap (current)', color='#C44E52', alpha=0.85)
bars2 = ax.bar(x + w/2, df['p2v1_10'], w, label='Stable Model — 10% cap (recommended)', color='#DD8452', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(LANES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Validation MAPE (%)', fontsize=11)
ax.set_title('Stable Model: 30% Cap vs 10% Cap — Validation MAPE by Lane\n(Lower is better)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, min(max(df['p2v1_30'].fillna(0).max(), df['p2v1_10'].fillna(0).max()) * 1.15, 120))
for bar in bars1:
    h = bar.get_height()
    if np.isfinite(h) and h < 120: ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=7, color='#C44E52')
for bar in bars2:
    h = bar.get_height()
    if np.isfinite(h) and h < 120: ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=7, color='#DD8452')
plt.tight_layout()
plt.savefig(f'{PLOT}/path2_30cap_vs_10cap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved path2_30cap_vs_10cap.png")

# ── Chart 3: Delta improvement chart ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors1 = ['#55A868' if v > 0 else '#C44E52' for v in df['p1_delta']]
axes[0].bar(LANES, df['p1_delta'], color=colors1, alpha=0.85)
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_xticklabels(LANES, rotation=45, ha='right', fontsize=9)
axes[0].set_ylabel('MAPE Improvement (pp)', fontsize=11)
axes[0].set_title('Model with COVID: v1.0 → v2.0\nMAPE Improvement per Lane (positive = v2 better)', fontsize=11, fontweight='bold')
axes[0].set_xticks(range(len(LANES)))

colors2 = ['#55A868' if v > 0 else '#C44E52' for v in df['p2_delta'].fillna(0)]
axes[1].bar(LANES, df['p2_delta'].fillna(0), color=colors2, alpha=0.85)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xticklabels(LANES, rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('MAPE Improvement (pp)', fontsize=11)
axes[1].set_title('Stable Model: 30% → 10% Cap\nMAPE Improvement per Lane (positive = 10% cap better)', fontsize=11, fontweight='bold')
axes[1].set_xticks(range(len(LANES)))
plt.tight_layout()
plt.savefig(f'{PLOT}/delta_improvements.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved delta_improvements.png")

# ── Summary stats ─────────────────────────────────────────────────────────────
valid1 = df['p1_delta'].dropna()
valid2 = df['p2_delta'].dropna()
print(f"\nPath 1 v1→v2: {(valid1>0).sum()}/{len(valid1)} lanes improved, avg gain {valid1.mean():.2f}pp, median {valid1.median():.2f}pp")
print(f"Path 2 30→10: {(valid2>0).sum()}/{len(valid2)} lanes improved, avg gain {valid2.mean():.2f}pp, median {valid2.median():.2f}pp")
print(f"\nPath 1 v1.0 median stk MAPE: {df['p1v1_stk'].median():.2f}%")
print(f"Path 1 v2.0 median stk MAPE: {df['p1v2_stk'].median():.2f}%")
print(f"Path 2 30% cap median MAPE:  {df['p2v1_30'].median():.2f}%")
print(f"Path 2 10% cap median MAPE:  {df['p2v1_10'].median():.2f}%")
