"""
Generate comparison charts for the model stacking report.
Charts produced:
  1. Grouped bar chart — MAPE by method per lane (Path 1 and Path 2)
  2. Method win-count summary
  3. Alpha/Beta weight scatter (constrained meta-learner)
  4. Per-lane validation line charts for selected lanes
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT   = '/home/ubuntu/fcl_forecast/outputs/antarctica'
PLOTS = '/home/ubuntu/fcl_forecast/outputs/meta_plots'
os.makedirs(PLOTS, exist_ok=True)

with open(os.path.join(OUT, 'meta_learner_results.json')) as f:
    results = json.load(f)
with open(os.path.join(OUT, 'meta_learner_weights.json')) as f:
    weights = json.load(f)

C = {
    'current':  '#1A3A5C',
    'opt_cap':  '#2E7D32',
    'meta_c':   '#F57C00',
    'meta_u':   '#D32F2F',
    'bg':       '#F8F9FA',
    'grid':     '#E0E0E0',
}

METHODS = ['mape_current_30pct', 'mape_optimal_cap', 'mape_meta_constrained', 'mape_meta_unconstrained']
METHOD_LABELS = ['Current\n(30% cap)', 'Optimal Cap\n(per-lane)', 'Meta-Learner\nConstrained', 'Meta-Learner\nUnconstrained']
METHOD_COLORS = [C['current'], C['opt_cap'], C['meta_c'], C['meta_u']]

# ── 1. Grouped bar chart per path ─────────────────────────────────────────
for path_tag in ['path1', 'path2']:
    path_label = 'Model with COVID' if path_tag == 'path1' else 'Stable Model'
    sub = {k: v for k, v in results.items() if v['path'] == path_tag}
    lanes = sorted(sub.keys(), key=lambda x: sub[x]['lane'])

    n = len(lanes)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_facecolor(C['bg'])
    fig.patch.set_facecolor('white')

    for i, (method, label, color) in enumerate(zip(METHODS, METHOD_LABELS, METHOD_COLORS)):
        vals = [sub[k][method] for k in lanes]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Mark best method per lane
    for j, k in enumerate(lanes):
        best_m = sub[k]['best_method']
        best_v = sub[k]['best_mape']
        ax.annotate('★', xy=(j + (METHODS.index(
            {'Current (30%)': 'mape_current_30pct',
             'Optimal Cap': 'mape_optimal_cap',
             'Meta Constrained': 'mape_meta_constrained',
             'Meta Unconstrained': 'mape_meta_unconstrained'}.get(best_m, 'mape_current_30pct')
        ) - 1.5) * width, best_v + 0.3),
                    ha='center', va='bottom', fontsize=9, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels([sub[k]['lane'] for k in lanes], fontsize=9)
    ax.set_ylabel('Validation MAPE (%)', fontsize=11)
    ax.set_title(f'{path_label} — Validation MAPE by Stacking Method\n(★ = best method per lane)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(True, axis='y', alpha=0.4, color=C['grid'])
    ax.set_axisbelow(True)

    plt.tight_layout()
    fp = os.path.join(PLOTS, f'mape_comparison_{path_tag}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fp}")

# ── 2. Method win-count summary (both paths side by side) ─────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Method Win Count — Which Stacking Method Achieves Lowest MAPE per Lane',
              fontsize=13, fontweight='bold')

win_labels = ['Current (30%)', 'Optimal Cap', 'Meta Constrained', 'Meta Unconstrained']
win_colors = [C['current'], C['opt_cap'], C['meta_c'], C['meta_u']]

for ax_i, (path_tag, path_label) in enumerate([('path1', 'Model with COVID'),
                                                ('path2', 'Stable Model')]):
    ax = axes2[ax_i]
    sub = {k: v for k, v in results.items() if v['path'] == path_tag}
    wins = {w: 0 for w in win_labels}
    for v in sub.values():
        wins[v['best_method']] = wins.get(v['best_method'], 0) + 1

    counts = [wins.get(w, 0) for w in win_labels]
    bars = ax.bar(win_labels, counts, color=win_colors, alpha=0.85, edgecolor='white')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title(path_label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Lanes Won', fontsize=10)
    ax.set_ylim(0, max(counts) + 2)
    ax.set_xticklabels(win_labels, rotation=15, ha='right', fontsize=9)
    ax.set_facecolor(C['bg'])
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fp2 = os.path.join(PLOTS, 'win_count_summary.png')
plt.savefig(fp2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fp2}")

# ── 3. Alpha/Beta weight scatter (constrained meta-learner) ───────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Constrained Meta-Learner Weights (α = Baseline Model, β = Adjustment Engine)\n'
              'α + β = 1 by construction', fontsize=12, fontweight='bold')

for ax_i, (path_tag, path_label) in enumerate([('path1', 'Model with COVID'),
                                                ('path2', 'Stable Model')]):
    ax = axes3[ax_i]
    sub_w = {k: v for k, v in weights.items() if v['path'] == path_tag}
    lanes_w = [v['lane'] for v in sub_w.values()]
    alphas  = [v['alpha_constrained'] for v in sub_w.values()]
    betas   = [v['beta_constrained']  for v in sub_w.values()]

    y_pos = range(len(lanes_w))
    ax.barh(y_pos, alphas, color=C['current'], alpha=0.8, label='α (Baseline Model)', height=0.4)
    ax.barh([y + 0.4 for y in y_pos], betas, color=C['meta_c'], alpha=0.8,
            label='β (Adjustment Engine)', height=0.4)

    ax.set_yticks([y + 0.2 for y in y_pos])
    ax.set_yticklabels(lanes_w, fontsize=9)
    ax.set_xlabel('Weight', fontsize=10)
    ax.set_title(path_label, fontsize=11, fontweight='bold')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    ax.set_facecolor(C['bg'])
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

plt.tight_layout()
fp3 = os.path.join(PLOTS, 'meta_weights.png')
plt.savefig(fp3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fp3}")

# ── 4. Average MAPE comparison bar chart (overall summary) ────────────────
fig4, ax4 = plt.subplots(figsize=(10, 5))
fig4.patch.set_facecolor('white')
ax4.set_facecolor(C['bg'])

path_labels = ['Model with COVID', 'Stable Model']
x4 = np.arange(len(path_labels))
width4 = 0.18

for i, (method, label, color) in enumerate(zip(METHODS, METHOD_LABELS, METHOD_COLORS)):
    avgs = []
    for path_tag in ['path1', 'path2']:
        sub = [v for v in results.values() if v['path'] == path_tag]
        avgs.append(np.mean([r[method] for r in sub]))
    ax4.bar(x4 + (i - 1.5) * width4, avgs, width4, label=label,
            color=color, alpha=0.85, edgecolor='white')
    for j, v in enumerate(avgs):
        ax4.text(x4[j] + (i - 1.5) * width4, v + 0.2, f'{v:.1f}%',
                 ha='center', va='bottom', fontsize=8)

ax4.set_xticks(x4)
ax4.set_xticklabels(path_labels, fontsize=11)
ax4.set_ylabel('Average Validation MAPE (%)', fontsize=11)
ax4.set_title('Average Validation MAPE by Method and Model\n(All 15 lanes, Jan 2025 – Mar 2026)',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax4.grid(True, axis='y', alpha=0.4)
ax4.set_axisbelow(True)

plt.tight_layout()
fp4 = os.path.join(PLOTS, 'avg_mape_summary.png')
plt.savefig(fp4, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fp4}")

print("\nAll charts saved.")
