#!/usr/bin/env python3
"""Figure 8: Training-set scaling of classification and reconstruction.

Modes:
  --source-svg FILE       Re-export approved SVG artwork (no data recalculation).
  --scaling SCALING.csv   Plot figure from the ten-run experiment results.

SCALING.csv columns:
 training_pct,seed,balanced_accuracy,weighted_f1,kappa,
 samplewise_spearman,featurewise_spearman

There should be 10 independent seeds per percentage (10,20,...,100).
The CSV should contain per-seed test metrics computed by the original
experiment pipeline. This script only aggregates mean ± SD and plots them;
model training and the computation of reconstruction correlations are upstream.

Examples:
 python regen_fig8_v3.py --source-svg 'Figure_8_VAE_Training_Set_Scaling(2).svg' --png
 python regen_fig8_v3.py --scaling scaling_experiment_results.csv
"""
from pathlib import Path
import argparse


def reexport(src: Path, dst: Path, png: bool, dpi: int):
    from lxml import etree
    if not src.is_file(): raise FileNotFoundError(src)
    tree=etree.parse(str(src)); ns='{http://www.w3.org/2000/svg}'
    n_text=0
    for el in tree.getroot().iter():
        if el.tag!=f'{ns}text':continue
        n_text+=1;el.set('font-family','Arial, Helvetica, sans-serif')
        value=''.join(el.itertext()).strip()
        if value in {'A','B','C','D','A.','B.','C.','D.'}:
            for node in el.iter():
                if node.text and node.text.strip()==value:
                    node.text=node.text.replace(value,value.lower(),1);break
            el.set('font-weight','bold')
    dst.parent.mkdir(parents=True,exist_ok=True)
    tree.write(str(dst),xml_declaration=True,encoding='utf-8')
    print(f'SVG artwork exported: {dst}; editable text elements={n_text}')
    if png:
        import cairosvg
        cairosvg.svg2png(url=str(dst),write_to=str(dst.with_suffix('.png')),scale=dpi/96)
        print(f'Rasterized: {dst.with_suffix(".png")}')


def plot(src: Path,outdir: Path,dpi: int):
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df=pd.read_csv(src)
    metrics=['balanced_accuracy','weighted_f1','kappa','samplewise_spearman','featurewise_spearman']
    expected=set(['training_pct','seed']+metrics)
    if not expected.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(expected-set(df.columns))}')
    if df.empty: raise ValueError('Scaling results are empty')
    df['training_pct']=df['training_pct'].astype(int)
    if not df.training_pct.isin(range(10,101,10)).all():
        raise ValueError('training_pct must be integer 10,20,...,100')
    for c in metrics:
        if df[c].isna().any() or not df[c].between(-1,1).all():
            raise ValueError(f'{c}: values must be finite correlations/scores in [-1,1]')
    if df.duplicated(['training_pct','seed']).any():
        raise ValueError('Duplicate training_pct/seed pairs')
    counts=df.groupby('training_pct')['seed'].nunique()
    missing_pct=set(range(10,101,10))-set(counts.index)
    if missing_pct: raise ValueError(f'Missing training percentages: {sorted(missing_pct)}')
    if any(counts!=10):
        print(f'WARNING: manuscript uses ten seeds/fraction; actual seeds: {counts.to_dict()}')
    avg=df.groupby('training_pct')[metrics].mean()
    sd=df.groupby('training_pct')[metrics].std(ddof=1).fillna(0)
    x=avg.index.to_numpy()
    fig,ax=plt.subplots(2,2,figsize=(12.7,8.9),sharex=True)
    settings=[('balanced_accuracy','a   Balanced accuracy','#179b70'),
              ('weighted_f1','b   Weighted F1 score','#2380bc'),
              ('kappa',"c   Cohen's κ",'#ec5149')]
    for axis,(col,title,color) in zip(ax.flat[:3],settings):
        y=avg[col].to_numpy();e=sd[col].to_numpy()
        axis.plot(x,y,color=color,lw=2,marker='o',markerfacecolor='white',markeredgewidth=1.8)
        axis.fill_between(x,y-e,y+e,color=color,alpha=.13)
        axis.errorbar(x,y,yerr=e,fmt='none',color=color,capsize=3)
        axis.set_title(title,loc='left',fontweight='bold')
        axis.set_xlabel('Training data (%)');axis.grid(axis='y',alpha=.22)
    axis=ax.flat[3]
    for col,label,color in [('samplewise_spearman','Sample-wise','#2380bc'),
                            ('featurewise_spearman','Feature-wise','#df8e29')]:
        y=avg[col].to_numpy();e=sd[col].to_numpy()
        axis.plot(x,y,color=color,marker='o',markerfacecolor='white',lw=2,label=label)
        axis.fill_between(x,y-e,y+e,color=color,alpha=.12)
        axis.errorbar(x,y,yerr=e,fmt='none',color=color,capsize=3)
    axis.set_title('d   Expression reconstruction',loc='left',fontweight='bold')
    axis.set_xlabel('Training data (%)')
    axis.set_ylabel('Median Spearman ρ')
    axis.legend(frameon=False)
    axis.grid(axis='y',alpha=.22)
    for a in ax.flat: a.set_xlim(7,103)
    fig.tight_layout(pad=2)
    outdir.mkdir(parents=True,exist_ok=True)
    for ext in ('svg','png'):
        out=outdir/f'Figure_8_NC.{ext}'
        fig.savefig(out,dpi=dpi,bbox_inches='tight')
        print('Saved:',out)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    grp=ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--source-svg',type=Path)
    grp.add_argument('--scaling',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('figures'))
    ap.add_argument('--png',action='store_true')
    ap.add_argument('--dpi',type=int,default=300)
    args=ap.parse_args()
    if args.dpi<72: ap.error('--dpi must be >= 72')
    if args.source_svg:
        reexport(args.source_svg,args.output_dir/'Figure_8_NC.svg',args.png,args.dpi)
    else:
        plot(args.scaling,args.output_dir,args.dpi)

if __name__=='__main__':
    main()
