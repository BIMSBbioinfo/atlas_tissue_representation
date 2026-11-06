#!/usr/bin/env python3
"""Figure 6: VAE versus BulkFormer on TARGET.

Two independent modes:
  --source-svg FILE     Reformat an EXISTING approved SVG master (artwork only).
  --metrics FILE        Plot from a summary CSV derived from sample predictions.

Summary CSV columns:
 model,overall_correct,overall_total,wilms_correct,wilms_brain,wilms_other
Rows must be 'Supervised VAE' and 'BulkFormer'. Each Wilms row must sum to 132.
This script does not infer tissue origin, run encoders or kNN; the --metrics
mode visualizes externally computed result summaries. For sample-level
reproducibility, preserve the model-inference and scoring pipelines separately.

Examples:
 python regen_fig6_v3.py --source-svg 'Fig6_VAE_vs_BulkFormer_TARGET(1).svg'
 python regen_fig6_v3.py --metrics fig6_metrics.csv --output-dir figures
"""
from pathlib import Path
import argparse
import re

SVG_MASTER = 'Fig6_VAE_vs_BulkFormer_TARGET(1).svg'


def export_svg_master(src: Path, dst: Path, png: bool, dpi: int) -> None:
    from lxml import etree
    if not src.is_file():
        raise FileNotFoundError(f'SVG source not found: {src}')
    root = etree.parse(str(src))
    ns = '{http://www.w3.org/2000/svg}'
    editable = 0
    for el in root.iter():
        if el.tag != f'{ns}text':
            continue
        editable += 1
        el.set('font-family', 'Arial, Helvetica, sans-serif')
        val = ''.join(el.itertext()).strip()
        # Modify panel labels only, NEVER quantitative annotations.
        if val in {'A', 'B', 'C', 'A.', 'B.', 'C.'}:
            for node in el.iter():
                if node.text and node.text.strip() == val:
                    node.text = node.text.replace(val, val.lower(), 1)
                    break
            el.set('font-weight', 'bold')
    dst.parent.mkdir(parents=True, exist_ok=True)
    root.write(str(dst), xml_declaration=True, encoding='utf-8')
    print(f'Saved approved-artwork re-export: {dst} (editable text elements: {editable})')
    if png:
        import cairosvg
        cairosvg.svg2png(url=str(dst), write_to=str(dst.with_suffix('.png')),
                         output_width=None, output_height=None, scale=dpi / 96)
        print(f'Saved: {dst.with_suffix(".png")} (rasterized from SVG at scale dpi/96)')


def plot_from_metrics(src: Path, output_dir: Path, dpi: int):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with src.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'model', 'overall_correct', 'overall_total',
                    'wilms_correct', 'wilms_brain', 'wilms_other'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing required columns: {sorted(required - set(reader.fieldnames or []))}')
        rows = {row['model'].strip(): row for row in reader}

    models = ['Supervised VAE', 'BulkFormer']
    if set(rows) != set(models):
        raise ValueError(f'Expected exactly two model rows named {models}; got {list(rows)}')
    numbers = {}
    for name in models:
        row = rows[name]
        n = {key: int(row[key]) for key in required if key != 'model'}
        if n['overall_total'] != 734:
            raise ValueError(f'{name}: expected 734 TARGET samples, got {n["overall_total"]}')
        if sum(n[k] for k in ('wilms_correct', 'wilms_brain', 'wilms_other')) != 132:
            raise ValueError(f'{name}: Wilms categories must sum to 132')
        if not all(v >= 0 for v in n.values()) or n['overall_correct'] > n['overall_total']:
            raise ValueError(f'{name}: invalid nonnegative count bounds')
        numbers[name] = n

    vae, bulk = (numbers[name] for name in models)
    a, b = vae['overall_correct']/734 * 100, bulk['overall_correct']/734 * 100
    fig = plt.figure(figsize=(11, 8))
    ax1 = fig.add_axes([.17, .61, .78, .33])
    teal, ochre, red, grey = '#24748c', '#c9872c', '#c13a2a', '#bbb8b2'
    ax1.bar([0, 1], [a, b], color=[teal, ochre], width=.55)
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.grid(axis='y', alpha=.2)
    ax1.set_axisbelow(True)
    ax1.set_xticks([0, 1], ['Supervised VAE-105M', 'BulkFormer-93M'])
    ax1.set_ylabel('Overall accuracy (%)')
    ax1.set_title('a. Developmental tissue-of-origin recovery on TARGET (n = 734)',
                  loc='left', fontweight='bold')
    for i, (val, color) in enumerate(zip([a, b], [teal, ochre])):
        ax1.text(i, val + 1.7, f'{val:.1f}%', ha='center', fontweight='bold',
                 color=color, fontsize=16)
    ax1.text(.5, 98, f'+{a-b:.1f} percentage points', ha='center', va='top')

    ax2 = fig.add_axes([.17, .15, .78, .33])
    y_vals = [1, 0]
    for y, n in zip(y_vals, [vae, bulk]):
        x = 0
        for k, color in [('wilms_correct', teal), ('wilms_brain', red), ('wilms_other', grey)]:
            value = n[k]
            ax2.barh(y, value, left=x, color=color, height=.60, edgecolor='white')
            if value >= 10:
                ax2.text(x + value/2, y, str(value), ha='center', va='center',
                         color='white' if k != 'wilms_other' else '#333333',
                         fontweight='bold')
            x += value
    ax2.set_yticks(y_vals, ['Supervised VAE', 'BulkFormer-93M'])
    ax2.set_xlim(0, 132)
    ax2.set_xlabel('Wilms tumour samples (n = 132)')
    ax2.set_title('b. Where each model maps Wilms tumour', loc='left', fontweight='bold')
    fig.legend(handles=[Patch(facecolor=teal, label='intermediate-mesoderm tissues (correct)'),
                        Patch(facecolor=red, label='brain (wrong germ layer)'),
                        Patch(facecolor=grey, label='other tissues')],
               loc='lower center', ncol=3, bbox_to_anchor=(.55, .015), frameon=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('svg', 'png'):
        path = output_dir / f'Figure_6_NC.{ext}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)
    print(f'Computed summary: VAE {vae["overall_correct"]}/734={a:.1f}%, BulkFormer {bulk["overall_correct"]}/734={b:.1f}%')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--source-svg', type=Path, metavar='FILE', help='Approved SVG artwork, no numeric recomputation')
    mode.add_argument('--metrics', type=Path, metavar='CSV', help='Author-verified Figure 6 summary counts')
    ap.add_argument('--output-dir', type=Path, default=Path('figures'))
    ap.add_argument('--png', action='store_true', help='Raster export in SVG artwork mode')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()
    if args.dpi < 72:
        ap.error('--dpi must be >= 72')
    if args.source_svg:
        export_svg_master(args.source_svg, args.output_dir/'Figure_6_NC.svg', args.png, args.dpi)
    else:
        plot_from_metrics(args.metrics, args.output_dir, args.dpi)

if __name__ == '__main__':
    main()
