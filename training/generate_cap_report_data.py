"""
Generate 10% vs 30% cap comparison data and charts for the report.
Produces:
  - cap_10_vs_30_results.csv   — per-lane MAPE at 10% and 30% cap
  - cap_comparison_path1.png   — grouped bar chart Path 1
  - cap_comparison_path2.png   — grouped bar chart Path 2
  - cap_comparison_delta.png   — MAPE improvement delta chart
  - cap_10_validation_*.csv    — per-lane validation predictions at 10% cap
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')

BASE  = '/home/ubuntu/fcl_forecast'
OUT   = os.path.join(BASE, 'outputs/antarctica')
MDL   = os.path.join(BASE, 'models/antarctica')
PLOTS = os.path.join(BASE, 'outputs/cap_report_plots')
os.makedirs(PLOTS, exist_ok=True)

C = {
    'cap30': '#1A3A5C',
    'cap10': '#2E7D32',
    'sar':   '#90A4AE',
    'delta': '#D32F2F',
    'bg':    '#F8F9FA',
}

def mape(actual, predicted):
    mask = (actual != 0) & (~np.isnan(actual)) & (~np.isnan(predicted))
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def apply_cap(sar, corr, cap_frac):
    max_c = np.abs(sar) * cap_frac
    return sar + np.clip(corr, -max_c, max_c)

records = []

for path_tag, bundle_file in [('path1', 'path1_final_models.pkl'),
                               ('path2', 'path2_final_models.pkl')]:
    bundle_fp = os.path.join(MDL, bundle_file)
    if not os.path.exists(bundle_fp):
        continue

    path_label = 'Model with COVID' if path_tag == 'path1' else 'Stable Model'
    print(f"\n{'='*60}")
    print(f"  {path_label}")
    print(f"{'='*60}")

    # Get lane list from bundle
    with open(bundle_fp, 'rb') as f:
        bundle = pickle.load(f)

    for lane in sorted(bundle.keys()):
        val_fp = os.path.join(OUT, f'{path_tag}_validation_{lane}.csv')
        if not os.path.exists(val_fp):
            continue
        val_df = pd.read_csv(val_fp, index_col=0, parse_dates=True)
        if not all(c in val_df.columns for c in ['actual', 'sarimax_pred', 'stacked_pred']):
            continue

        actual  = val_df['actual'].values.astype(float)
        sar     = val_df['sarimax_pred'].values.astype(float)
        stk30   = val_df['stacked_pred'].values.astype(float)
        corr30  = stk30 - sar

        # 30% cap (current) — use stacked_pred directly
        mape_sar   = mape(actual, sar)
        mape_30    = mape(actual, stk30)

        # 10% cap — re-apply to same raw correction
        stk10 = apply_cap(sar, corr30, 0.10)
        mape_10 = mape(actual, stk10)

        delta = mape_30 - mape_10  # positive = 10% is better

        records.append({
            'path': path_tag,
            'path_label': path_label,
            'lane': lane,
            'n_val': len(val_df),
            'mape_baseline': round(mape_sar, 2),
            'mape_cap30': round(mape_30, 2),
            'mape_cap10': round(mape_10, 2),
            'delta_pp': round(delta, 2),
            'winner': '10% Cap' if mape_10 < mape_30 else ('30% Cap' if mape_30 < mape_10 else 'Tie'),
        })

        # Save per-lane 10% cap validation
        val_out = val_df[['actual', 'sarimax_pred', 'stacked_pred']].copy()
        val_out.columns = ['actual', 'baseline_pred', 'cap30_pred']
        val_out['cap10_pred'] = stk10
        val_out['cap30_error_pct'] = ((val_out['cap30_pred'] - val_out['actual']) / val_out['actual'] * 100).round(2)
        val_out['cap10_error_pct'] = ((val_out['cap10_pred'] - val_out['actual']) / val_out['actual'] * 100).round(2)
        val_out.to_csv(os.path.join(OUT, f'cap10_validation_{path_tag}_{lane}.csv'))

        print(f"  {lane:8s} | Baseline={mape_sar:6.2f}%  30%={mape_30:6.2f}%  10%={mape_10:6.2f}%  Δ={delta:+.2f}pp  → {records[-1]['winner']}")

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT, 'cap_10_vs_30_results.csv'), index=False)
print(f"\nSaved: cap_10_vs_30_results.csv")

# ── Charts ────────────────────────────────────────────────────────────────

for path_tag in ['path1', 'path2']:
    sub = df[df['path'] == path_tag].copy()
    path_label = sub['path_label'].iloc[0]
    lanes = sub['lane'].values
    n = len(lanes)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.set_facecolor(C['bg'])
    fig.patch.set_facecolor('white')

    b1 = ax.bar(x - w,     sub['mape_baseline'].values, w, label='Baseline Model Only', color=C['sar'],   alpha=0.75, edgecolor='white')
    b2 = ax.bar(x,         sub['mape_cap30'].values,    w, label='30% Cap (Current)',   color=C['cap30'], alpha=0.85, edgecolor='white')
    b3 = ax.bar(x + w,     sub['mape_cap10'].values,    w, label='10% Cap',             color=C['cap10'], alpha=0.85, edgecolor='white')

    # Annotate winner
    for i, row in sub.reset_index(drop=True).iterrows():
        if row['winner'] == '10% Cap':
            ax.annotate('★', xy=(i + w, row['mape_cap10'] + 0.3), ha='center', fontsize=10, color=C['cap10'])
        elif row['winner'] == '30% Cap':
            ax.annotate('★', xy=(i,     row['mape_cap30'] + 0.3), ha='center', fontsize=10, color=C['cap30'])

    ax.set_xticks(x)
    ax.set_xticklabels(lanes, fontsize=9)
    ax.set_ylabel('Validation MAPE (%)', fontsize=11)
    ax.set_title(f'{path_label} — Validation MAPE: Baseline vs 30% Cap vs 10% Cap\n'
                 f'(★ = winning cap per lane, Jan 2025 – Mar 2026)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(True, axis='y', alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fp = os.path.join(PLOTS, f'cap_comparison_{path_tag}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fp}")

# Delta chart — improvement from switching 30% → 10%
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))
fig2.suptitle('MAPE Improvement from Tightening Adjustment Engine Cap: 30% → 10%\n'
              '(Positive = 10% cap is better; Negative = 30% cap is better)',
              fontsize=13, fontweight='bold')

for ax_i, path_tag in enumerate(['path1', 'path2']):
    ax = axes2[ax_i]
    sub = df[df['path'] == path_tag].sort_values('delta_pp', ascending=True)
    path_label = sub['path_label'].iloc[0]
    colors = [C['cap10'] if v >= 0 else C['cap30'] for v in sub['delta_pp']]
    bars = ax.barh(sub['lane'], sub['delta_pp'], color=colors, alpha=0.85, edgecolor='white', height=0.6)

    for bar, (_, row) in zip(bars, sub.iterrows()):
        x_pos = row['delta_pp']
        label = f"{x_pos:+.2f}pp  ({row['mape_cap30']:.1f}% → {row['mape_cap10']:.1f}%)"
        ax.text(x_pos + (0.1 if x_pos >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                label, va='center', ha='left' if x_pos >= 0 else 'right', fontsize=8)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('MAPE Improvement (pp)', fontsize=10)
    ax.set_title(path_label, fontsize=11, fontweight='bold')
    ax.set_facecolor(C['bg'])
    ax.grid(True, axis='x', alpha=0.3)

    # Summary stats
    wins_10 = (sub['delta_pp'] > 0).sum()
    wins_30 = (sub['delta_pp'] < 0).sum()
    avg_gain = sub['delta_pp'].mean()
    ax.set_xlabel(f'MAPE Improvement (pp)  |  10% wins {wins_10} lanes, 30% wins {wins_30} lanes  |  Avg gain: {avg_gain:+.2f}pp', fontsize=9)

plt.tight_layout()
fp2 = os.path.join(PLOTS, 'cap_comparison_delta.png')
plt.savefig(fp2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fp2}")

# Print summary
print("\n" + "="*65)
for path_tag in ['path1', 'path2']:
    sub = df[df['path'] == path_tag]
    label = sub['path_label'].iloc[0]
    print(f"\n{label}")
    print(f"  Avg MAPE — Baseline only:  {sub['mape_baseline'].mean():.2f}%")
    print(f"  Avg MAPE — 30% cap:        {sub['mape_cap30'].mean():.2f}%")
    print(f"  Avg MAPE — 10% cap:        {sub['mape_cap10'].mean():.2f}%")
    print(f"  Avg gain (30→10):          {sub['delta_pp'].mean():+.2f}pp")
    print(f"  Lanes where 10% wins:      {(sub['delta_pp'] > 0).sum()} / {len(sub)}")
    print(f"  Lanes where 30% wins:      {(sub['delta_pp'] < 0).sum()} / {len(sub)}")
print("="*65)
print("\nDone.")
