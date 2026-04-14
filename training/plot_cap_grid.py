"""
Plot cap grid search results — MAPE curves per lane and summary heatmap.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

OUT   = '/home/ubuntu/fcl_forecast/outputs/antarctica'
PLOTS = '/home/ubuntu/fcl_forecast/outputs/cap_grid_plots'
os.makedirs(PLOTS, exist_ok=True)

res_df  = pd.read_csv(os.path.join(OUT, 'cap_grid_search_results.csv'))
sum_df  = pd.read_csv(os.path.join(OUT, 'cap_grid_summary.csv'))

# Cap order for x-axis
CAP_ORDER = ['5%','10%','15%','20%','25%','30%','40%','50%','75%','100%','uncapped','uncapped*']
CAP_LABELS = ['5%','10%','15%','20%','25%','30%','40%','50%','75%','100%','Uncapped']

COLORS = {
    'path1': '#1A3A5C',
    'path2': '#2E7D32',
    'highlight': '#D32F2F',
    'current': '#F57C00',
}

# ── 1. Per-path MAPE curves (one subplot per lane) ─────────────────────────
for path_tag in ['path1', 'path2']:
    sub = res_df[res_df['path'] == path_tag].copy()
    lanes = sorted(sub['lane'].unique())
    n = len(lanes)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 3.8))
    fig.suptitle(
        f'Cap Grid Search — {"Model with COVID" if path_tag=="path1" else "Stable Model"}\n'
        f'Validation MAPE (Jan 2025 – Mar 2026) vs XGBoost Correction Cap',
        fontsize=14, fontweight='bold', y=1.01
    )
    axes_flat = axes.flatten() if nrows > 1 else axes

    for idx, lane in enumerate(lanes):
        ax = axes_flat[idx]
        lane_df = sub[sub['lane'] == lane].copy()

        # Map cap to numeric order
        lane_df['cap_order'] = lane_df['cap'].map(
            {c: i for i, c in enumerate(CAP_ORDER)}
        )
        lane_df = lane_df.dropna(subset=['cap_order']).sort_values('cap_order')

        x = range(len(lane_df))
        y = lane_df['val_mape'].values
        caps = lane_df['cap'].values

        ax.plot(x, y, color=COLORS[path_tag], linewidth=2, marker='o', markersize=5)

        # Highlight current 30% cap
        cur_idx = list(caps).index('30%') if '30%' in caps else None
        if cur_idx is not None:
            ax.axvline(cur_idx, color=COLORS['current'], linestyle='--', linewidth=1.2, alpha=0.8)
            ax.plot(cur_idx, y[cur_idx], 'o', color=COLORS['current'], markersize=8, zorder=5)

        # Highlight best cap
        best_idx = int(np.argmin(y))
        ax.plot(best_idx, y[best_idx], '*', color=COLORS['highlight'], markersize=12, zorder=6)

        ax.set_title(f'{lane}', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(caps)))
        ax.set_xticklabels([c.replace('uncapped*','Unc*').replace('uncapped','Unc') for c in caps],
                           rotation=45, fontsize=7)
        ax.set_ylabel('MAPE (%)', fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F8F9FA')

        # Annotate best
        best_cap = caps[best_idx]
        best_mape = y[best_idx]
        ax.annotate(f'Best: {best_cap}\n{best_mape:.1f}%',
                    xy=(best_idx, best_mape),
                    xytext=(best_idx + 0.5, best_mape + (y.max()-y.min())*0.15 + 0.1),
                    fontsize=7, color=COLORS['highlight'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=1))

    # Hide unused subplots
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['current'], linestyle='--', linewidth=1.5, label='Current cap (30%)'),
        Line2D([0], [0], marker='*', color=COLORS['highlight'], linestyle='None', markersize=10, label='Optimal cap'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_fp = os.path.join(PLOTS, f'cap_grid_{path_tag}.png')
    plt.savefig(out_fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_fp}")

# ── 2. Summary heatmap — optimal cap and gain per lane ─────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
fig2.suptitle('Cap Grid Search Summary — Optimal Cap & MAPE Improvement vs 30% Baseline',
              fontsize=13, fontweight='bold')

for ax_idx, (path_tag, path_label) in enumerate([('path1', 'Model with COVID'),
                                                   ('path2', 'Stable Model')]):
    ax = axes2[ax_idx]
    sub = sum_df[sum_df['Path'] == path_tag].copy()
    sub = sub.sort_values('Lane')

    # Extract numeric improvement
    sub['imp_num'] = sub['Improvement'].str.replace('pp','').astype(float)
    sub['opt_cap_num'] = sub['Optimal Cap'].str.replace('%','').replace('uncapped','999').replace('uncapped*','999')

    lanes   = sub['Lane'].values
    imp     = sub['imp_num'].values
    opt_cap = sub['Optimal Cap'].values
    cur_mape = sub['Current Cap (30%) MAPE'].str.replace('%','').astype(float).values
    opt_mape = sub['Optimal MAPE'].str.replace('%','').astype(float).values

    y_pos = range(len(lanes))
    colors_bar = ['#2E7D32' if v > 0 else '#1A3A5C' for v in imp]

    bars = ax.barh(y_pos, imp, color=colors_bar, alpha=0.85, edgecolor='white', height=0.6)

    # Annotate with optimal cap and MAPEs
    for i, (bar, lane, oc, cm, om, iv) in enumerate(zip(bars, lanes, opt_cap, cur_mape, opt_mape, imp)):
        ax.text(max(iv + 0.05, 0.1), i,
                f'Best: {oc}  ({om:.1f}% vs {cm:.1f}%)',
                va='center', fontsize=8, color='#0A1628')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(lanes, fontsize=9)
    ax.set_xlabel('MAPE Improvement (pp) vs 30% Cap', fontsize=10)
    ax.set_title(path_label, fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_facecolor('#F8F9FA')
    ax.invert_yaxis()

plt.tight_layout()
out_fp2 = os.path.join(PLOTS, 'cap_grid_summary.png')
plt.savefig(out_fp2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_fp2}")

# ── 3. Print clean summary table ───────────────────────────────────────────
print("\n" + "="*80)
print(f"{'Path':<8} {'Lane':<8} {'30% MAPE':>10} {'Optimal Cap':>12} {'Optimal MAPE':>13} {'Gain':>8}")
print("-"*80)
for _, row in sum_df.sort_values(['Path','Lane']).iterrows():
    print(f"{row['Path']:<8} {row['Lane']:<8} {row['Current Cap (30%) MAPE']:>10} "
          f"{row['Optimal Cap']:>12} {row['Optimal MAPE']:>13} {row['Improvement']:>8}")
print("="*80)

# Global recommendation
p1 = sum_df[sum_df['Path']=='path1']['imp_num'] if 'imp_num' in sum_df.columns else \
     sum_df[sum_df['Path']=='path1']['Improvement'].str.replace('pp','').astype(float)
p2 = sum_df[sum_df['Path']=='path2']['Improvement'].str.replace('pp','').astype(float)
print(f"\nPath1 avg gain from optimal cap: {sum_df[sum_df['Path']=='path1']['Improvement'].str.replace('pp','').astype(float).mean():+.2f}pp")
print(f"Path2 avg gain from optimal cap: {sum_df[sum_df['Path']=='path2']['Improvement'].str.replace('pp','').astype(float).mean():+.2f}pp")
print("\nDone.")
