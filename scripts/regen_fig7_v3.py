#!/usr/bin/env python3
"""Figure 7: HPA transfer and DepMap fine-tuning.

Modes:
  --source-svg FILE             Re-export existing author-approved SVG artwork.
  --hpa HPA.csv --depmap D.csv  Plot Figure 7 from independent result files.

HPA.csv: one row per evaluable profile (expected 385); columns:
 true_tissue, vae_top5, pca121_pred, all_genes_pred, top2000_hvg_pred
 'vae_top5' is five ranked tissue labels separated by semicolons (;) with
 best prediction first. Tissue labels must use the same controlled vocabulary.

D.csv: DepMap performance per subsampling run; columns:
 method,n_labelled,run,macro_f1
 method values: 'Pretrained + fine-tuned VAE', 'VAE trained de novo', 'Random forest'
 n_labelled values include 0 only for the zero-shot pretrained VAE.

The --source-svg option preserves approved results WITHOUT recomputation.
The --hpa/--depmap option computes descriptive rates/means/s.d. from supplied
results; it does not run HPA inference, kNN fitting, or model fine-tuning.

Examples:
 python regen_fig7_v3.py --source-svg Figure7_transfer_across_expression_contexts.svg --png
 python regen_fig7_v3.py --hpa hpa_predictions.csv --depmap depmap_runs.csv
"""
from pathlib import Path
import argparse


def reexport(src: Path, dst: Path, png: bool, dpi: int):
    from lxml import etree
    if not src.is_file():
        raise FileNotFoundError(src)
    tree = etree.parse(str(src))
    ns = '{http://www.w3.org/2000/svg}'
    n_text = 0
    for el in tree.getroot().iter():
        if el.tag != f'{ns}text':
            continue
        n_text += 1
        el.set('font-family', 'Arial, Helvetica, sans-serif')
        value = ''.join(el.itertext()).strip()
        if value in {'A', 'B', 'C', 'A.', 'B.', 'C.'}:
            for node in el.iter():
                if node.text and node.text.strip() == value:
                    node.text = node.text.replace(value, value.lower(), 1)
                    break
            el.set('font-weight', 'bold')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dst), xml_declaration=True, encoding='utf-8')
    print(f'SVG artwork exported: {dst}; {n_text} editable text elements')
    if png:
        import cairosvg
        cairosvg.svg2png(url=str(dst), write_to=str(dst.with_suffix('.png')), scale=dpi/96)
        print(f'Rasterized: {dst.with_suffix(".png")}')


def plot(hpa_path: Path, depmap_path: Path, outdir: Path, dpi: int):
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hpa = pd.read_csv(hpa_path, keep_default_na=False)
    cols = {'true_tissue', 'vae_top5', 'pca121_pred', 'all_genes_pred', 'top2000_hvg_pred'}
    if not cols.issubset(hpa.columns):
        raise ValueError(f'HPA input missing: {sorted(cols-set(hpa.columns))}')
    if hpa.empty:
        raise ValueError('HPA input is empty')
    if len(hpa) != 385:
        print(f'WARNING: manuscript Figure 7 uses 385 HPA profiles; received {len(hpa)}')
    true = hpa['true_tissue'].astype(str).str.strip()
    ranks = [list(map(str.strip, str(x).split(';'))) for x in hpa['vae_top5']]
    if any(len(xs) < 5 for xs in ranks):
        raise ValueError('Each vae_top5 cell must list at least 5 ranked tissue labels separated by ;')
    topk = np.array([np.mean([t in row[:k] for t,row in zip(true, ranks)]) * 100 for k in (1,2,3,5)])
    base = [np.mean(hpa[col].astype(str).str.strip().eq(true)) * 100
            for col in ('pca121_pred','all_genes_pred','top2000_hvg_pred')]
    depmap = pd.read_csv(depmap_path)
    dep_cols = {'method','n_labelled','run','macro_f1'}
    if not dep_cols.issubset(depmap.columns):
        raise ValueError(f'DepMap input missing: {sorted(dep_cols-set(depmap.columns))}')
    known = {'Pretrained + fine-tuned VAE', 'VAE trained de novo', 'Random forest'}
    if set(depmap['method'].unique())-known:
        raise ValueError(f'Unrecognized method labels: {set(depmap.method.unique())-known}')
    if depmap['macro_f1'].isna().any() or not depmap['macro_f1'].between(0,1).all():
        raise ValueError('macro_f1 must be numeric and lie between 0 and 1')
    depmap['n_labelled'] = depmap['n_labelled'].astype(int)
    summary = depmap.groupby(['method','n_labelled'], as_index=False)['macro_f1'].agg(['mean','std']).reset_index()
    summary['std'] = summary['std'].fillna(0)

    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={'width_ratios':[1,1.05,1.65]})
    teal = '#2c9d92'
    bars = [topk[0]] + base
    axs[0].bar(range(4),bars,color=[teal,'#4e83a4','#788189','#b2b7bc'],width=.67)
    axs[0].set_xticks(range(4),['Pretrained\nVAE','PCA(121)\n+ kNN','All genes\n+ kNN','Top-2000 HVG\n+ kNN'])
    axs[0].set_ylabel('Top-1 tissue accuracy (%)')
    axs[0].set_ylim(0,max(55,max(bars)+8))
    for x,y in enumerate(bars): axs[0].text(x,y+1,f'{y:.1f}',ha='center',fontsize=9)
    axs[0].set_title('a   HPA tissue-of-origin prediction',loc='left')
    axs[1].plot([1,2,3,5],topk,color=teal,marker='o',lw=2)
    axs[1].set_xticks([1,2,3,5],['Top-1','Top-2','Top-3','Top-5'])
    axs[1].set_ylabel('Correct tissue recovered (%)')
    axs[1].set_ylim(max(0,min(topk)-7),min(100,max(topk)+6))
    for x,y in zip([1,2,3,5],topk): axs[1].text(x,y+1.2,f'{y:.1f}',ha='center',fontsize=9)
    axs[1].set_title('b   HPA ranked tissue recovery',loc='left')
    settings = [('Pretrained + fine-tuned VAE', teal, 'o'),
                ('VAE trained de novo', '#eb7056', 's'),
                ('Random forest', '#77818b', '^')]
    for name, color, mark in settings:
        subset=summary[summary['method']==name].sort_values('n_labelled')
        if subset.empty: continue
        axs[2].errorbar(subset['n_labelled'],subset['mean'],yerr=subset['std'],
                        color=color,marker=mark,markersize=6,linewidth=2,capsize=3,label=name)
    axs[2].set_xlabel('Labelled cell-line training samples')
    axs[2].set_ylabel('Macro-F1')
    axs[2].legend(loc='lower right',fontsize=8,frameon=False)
    axs[2].set_title('c   Cell-line transfer and fine-tuning',loc='left')
    for ax in axs:
        ax.grid(axis='y',alpha=.22)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    outdir.mkdir(parents=True,exist_ok=True)
    for ext in ('svg','png'):
        dst=outdir/f'Figure_7_NC.{ext}'
        fig.savefig(dst,dpi=dpi,bbox_inches='tight')
        print(f'Saved: {dst}')
    plt.close(fig)
    print('Computed HPA Top-k:',dict(zip([1,2,3,5],np.round(topk,2))))


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source-svg',type=Path,help='Existing approved SVG master; artwork only')
    ap.add_argument('--hpa',type=Path,help='HPA prediction table (385 rows)')
    ap.add_argument('--depmap',type=Path,help='DepMap metrics per subsampling run')
    ap.add_argument('--output-dir',type=Path,default=Path('figures'))
    ap.add_argument('--png',action='store_true',help='Also rasterize SVG in artwork mode')
    ap.add_argument('--dpi',type=int,default=300)
    args=ap.parse_args()
    if args.dpi<72: ap.error('--dpi must be >= 72')
    if args.source_svg and (args.hpa or args.depmap): ap.error('Choose EITHER --source-svg OR --hpa + --depmap')
    if args.source_svg:
        reexport(args.source_svg,args.output_dir/'Figure_7_NC.svg',args.png,args.dpi)
    elif args.hpa and args.depmap:
        plot(args.hpa,args.depmap,args.output_dir,args.dpi)
    else:
        ap.error('Provide --source-svg OR both --hpa and --depmap')

if __name__=='__main__':
    main()
