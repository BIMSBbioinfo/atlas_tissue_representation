#!/usr/bin/env python3
"""Figure 5: Benchmark comparison — v3 (118k Scope B)"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── v3 numbers (from baseline_hvg_knn_v3.py + classification_metrics_v3.json) ──
methods = [
    'scGPT\n(zero-shot)',
    'HVG+UMAP\n+kNN',
    'HVG+PCA(50)\n+kNN',
    'All genes\n+PCA(121)+kNN',
    'All genes\n+kNN',
    'Top-2000\nHVG+kNN',
    'Supervised\nVAE [v3]',
]
accuracies = [61.0, 84.2, 89.6, 92.0, 92.6, 93.4, 94.9]

colors = {
    'scGPT\n(zero-shot)':         '#E74C3C',
    'HVG+UMAP\n+kNN':             '#888888',
    'HVG+PCA(50)\n+kNN':          '#888888',
    'All genes\n+PCA(121)+kNN':   '#888888',
    'All genes\n+kNN':            '#2980B9',
    'Top-2000\nHVG+kNN':          '#2980B9',
    'Supervised\nVAE [v3]':       '#27AE60',
}

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={'width_ratios': [1.6, 1]})

# Panel A: bar chart
x    = np.arange(len(methods))
bars = ax1.bar(x, accuracies, color=[colors[m] for m in methods],
               width=0.65, zorder=3)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
             f'{acc:.1f}%', ha='center', va='bottom',
             fontsize=9.5, fontweight='bold')

ax1.axhline(y=94.9, color='#27AE60', linestyle='--', linewidth=1.2,
            alpha=0.7, zorder=2)

ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=9)
ax1.set_ylabel('Balanced Accuracy (%)', fontsize=11)
ax1.set_ylim(0, 100)
ax1.set_yticks(range(0, 101, 20))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.3, zorder=1)

legend_patches = [
    mpatches.Patch(color='#E74C3C', label='Foundation model (zero-shot)'),
    mpatches.Patch(color='#888888', label='Unsupervised reduction + kNN'),
    mpatches.Patch(color='#2980B9', label='Direct kNN on expression'),
    mpatches.Patch(color='#27AE60', label='Supervised VAE (this work, v3)'),
]
ax1.legend(handles=legend_patches, fontsize=8.5, loc='upper left',
           frameon=True, framealpha=0.9)
ax1.text(-0.04, 1.04, 'A', transform=ax1.transAxes,
         fontsize=16, fontweight='bold')

# Panel B: accuracy vs dimensionality scatter
scatter_data = [
    (768,    61.0, '#E74C3C', 'scGPT (zero-shot)'),
    (50,     84.2, '#888888', None),
    (50,     89.6, '#888888', None),
    (121,    92.0, '#888888', None),
    (2000,   93.4, '#2980B9', None),
    (16292,  92.6, '#2980B9', 'All genes +kNN'),
    (121,    94.9, '#27AE60', 'Supervised VAE [v3]'),
]
for dim, acc, col, label in scatter_data:
    ax2.scatter(dim, acc, color=col, s=120, zorder=3,
                edgecolors='white', linewidths=0.8)

ax2.annotate('Supervised VAE [v3]\n(this work)',
             xy=(121, 94.9), xytext=(220, 92.5),
             fontsize=8.5, color='#27AE60',
             arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.2))
ax2.annotate('scGPT (zero-shot)',
             xy=(768, 61.0), xytext=(400, 64),
             fontsize=8.5, color='#E74C3C',
             arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.2))
ax2.annotate('All genes +kNN',
             xy=(16292, 92.6), xytext=(5000, 95),
             fontsize=8.5, color='#2980B9',
             arrowprops=dict(arrowstyle='->', color='#2980B9', lw=1.2))

ax2.annotate('135× compression\nhigher accuracy',
             xy=(121, 94.9), xytext=(220, 82),
             fontsize=8, color='#27AE60',
             arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.0,
                             connectionstyle='arc3,rad=0.2'))

ax2.set_xscale('log')
ax2.set_xlabel('Representation Dimensionality', fontsize=11)
ax2.set_ylabel('Balanced Accuracy (%)', fontsize=11)
ax2.set_xlim(30, 30000)
ax2.set_ylim(55, 98)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(alpha=0.3, zorder=1)
ax2.text(-0.08, 1.04, 'B', transform=ax2.transAxes,
         fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('Fig5_benchmarks_v3_118k.svg', dpi=300, bbox_inches='tight')
plt.savefig('Fig5_benchmarks_v3_118k.png', dpi=300, bbox_inches='tight')
print("Saved: Fig5_benchmarks_v3_118k.svg / .png")
print(f"\nv3 Key numbers:")
print(f"  Supervised VAE:    94.9%  (dim=121)")
print(f"  Top-2000 HVG+kNN:  93.4%  (dim=2000)")
print(f"  All genes+kNN:     92.6%  (dim=16292)")
print(f"  scGPT zero-shot:   61.0%  (dim=768)")
