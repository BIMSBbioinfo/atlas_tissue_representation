import os, re, json, glob
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
FULL = 118263
rows = []
for f in sorted(glob.glob('stats/n*_s*.stats.csv')):
    m = re.match(r'stats/n(\d+)_s(\d+)\.stats\.csv$', f); N, S = int(m[1]), int(m[2])
    p = f'stats/n{N}_s{S}'
    if not os.path.exists(p + '.timing.txt'): continue
    r = {'N': N, 'seed': S}
    for _, x in pd.read_csv(f).iterrows(): r[x['metric']] = x['value']
    r['epochs_trained'] = json.load(open(p + '.final_model_config.json')).get('current_epoch')
    r['wallclock_min'] = int(re.search(r'wallclock_s=(\d+)', open(p + '.timing.txt').read())[1]) / 60
    rows.append(r)
df = pd.DataFrame(rows).sort_values(['N', 'seed'])
df['pct'] = (np.round(df.N / 11800) * 10).astype(int)
df.to_csv('scaling_results_all_runs.csv', index=False)
mets = ['balanced_acc', 'f1_score', 'kappa', 'average_auroc', 'average_aupr', 'mse',
        'sample_spearman_corr_mean', 'feature_spearman_corr_mean',
        'sample_spearman_corr_median', 'feature_spearman_corr_median',
        'sample_corr_mean', 'feature_corr_mean', 'sample_corr_median', 'feature_corr_median', 'epochs_trained', 'wallclock_min']
g = df.groupby(['pct', 'N'])[mets]
summ = g.agg(['mean', 'std']); summ.columns = ['_'.join(c) for c in summ.columns]
summ['n_seeds'] = df.groupby(['pct', 'N']).size()
summ.to_csv('scaling_results_summary.csv')
print(summ.round(4).to_string())

# figure source data (Excel): per-size mean/SD table + per-run values
LAB = {'balanced_acc': 'Balanced accuracy', 'f1_score': 'F1 (weighted)', 'kappa': "Cohen's kappa", 'average_auroc': 'AUROC (avg)',
       'average_aupr': 'AUPR (avg)', 'mse': 'Reconstruction MSE',
       'sample_spearman_corr_mean': 'Spearman per sample (mean)', 'sample_spearman_corr_median': 'Spearman per sample (median)',
       'feature_spearman_corr_mean': 'Spearman per gene (mean)', 'feature_spearman_corr_median': 'Spearman per gene (median)',
       'sample_corr_mean': 'Pearson per sample (mean)', 'sample_corr_median': 'Pearson per sample (median)',
       'feature_corr_mean': 'Pearson per gene (mean)', 'feature_corr_median': 'Pearson per gene (median)',
       'epochs_trained': 'Epochs trained', 'wallclock_min': 'Wall-clock (min)'}
src = summ.reset_index().rename(columns={'pct': 'Training set (%)', 'N': 'Training samples', 'n_seeds': 'Seeds'})
src = src.rename(columns={f'{k}_{a}': f'{v} {"SD" if a == "std" else a}' for k, v in LAB.items() for a in ('mean', 'std')})
runs = df[['pct', 'N', 'seed'] + mets].rename(columns={'pct': 'Training set (%)', 'N': 'Training samples', 'seed': 'Seed', **LAB})
with pd.ExcelWriter('figure source data.xlsx') as xw:
    src.to_excel(xw, sheet_name='Summary (mean, SD over seeds)', index=False)
    runs.to_excel(xw, sheet_name='Per-run', index=False)

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
C = ['#2a78d6', '#eb6834', '#1baf7a']
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
def panel(ax, series, title, ylabel, loc):
    ax.set_facecolor(SURF)
    for (col, lab), c in zip(series, C):
        gg = df.groupby('pct')[col]; mu, sd = gg.mean(), gg.std()
        ax.errorbar(mu.index, mu, yerr=sd, color=c, lw=2, marker='o', ms=7, mec=SURF, mew=1.5,
                    capsize=3, elinewidth=1.2, label=lab)
    ax.set_title(title, loc='left', color=INK, fontsize=12, fontweight='bold')
    ax.set_xlabel('Training set size (% of 118,263 samples)', color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.set_xticks(range(10, 101, 10)); ax.set_xticklabels([f'{p}%' for p in range(10, 101, 10)])
    ax.grid(axis='y', color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    if len(series) > 1: ax.legend(frameon=False, loc=loc, labelcolor=INK2)
panel(axs[0], [('f1_score', 'Weighted F1')], 'A  Tissue classification', 'Weighted F1 score (test)', 'lower right')
panel(axs[1], [('sample_spearman_corr_median', 'Sample-wise'), ('feature_spearman_corr_median', 'Feature-wise')],
      'B  Expression reconstruction', 'Median Spearman correlation (test)', 'lower right')
fig.tight_layout(); fig.savefig('scaling_figure.png', dpi=200, facecolor=SURF)
