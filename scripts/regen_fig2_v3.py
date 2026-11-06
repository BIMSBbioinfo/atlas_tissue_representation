#!/usr/bin/env python3
"""Figure 2 — final 118k Scope B held-out tissue accuracy.

Runs the existing supervised-VAE classifier on the held-out HDF5 test split,
reports balanced accuracy / weighted F1, and generates the Figure 2 bar chart.
The plotting block follows the approved Nature Communications artwork:
41 evaluated classes, descending top-to-bottom order, sample-size colours,
and a balanced-accuracy reference line. No results are hard-coded.
"""
import glob
import os
import sys
import json

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, '.')
from h5_dataloader import H5DataImporter

# ----------------------------------------------------------------------
# CONFIG — original v3 paths retained
# ----------------------------------------------------------------------
RESULTS_DIR = 'results_denoising_vae_411k_B'
MODEL_PATH  = f'{RESULTS_DIR}/standard_vae.pth'
CKPT_GLOB   = f'{RESULTS_DIR}/ckpt_standard/best_epoch=*.ckpt'
DATA_PATH   = 'processed_scaled_411k_tissue_B_h5'

OUT_SVG = 'Fig2_per_class_accuracy_v3_118k.svg'
OUT_PNG = 'Fig2_per_class_accuracy_v3_118k.png'
OUT_CSV = 'fig2_perclass_v3_118k.csv'
OUT_JSON = f'{RESULTS_DIR}/classification_metrics_v3.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------
# Load the standard VAE and its best checkpoint (original logic)
# ----------------------------------------------------------------------
print(f'[1] Loading Standard VAE from {MODEL_PATH}...')
model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = model.to(device)

ckpts = glob.glob(CKPT_GLOB)
if not ckpts:
    sys.exit(f'ERROR: No checkpoints found at {CKPT_GLOB}')
best_ckpt = sorted(
    ckpts,
    key=lambda p: float(p.split('val_loss=')[1].split('.ckpt')[0])
)[0]
print(f'    Best checkpoint: {os.path.basename(best_ckpt)}')
ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f'    Model loaded, parameters: {sum(p.numel() for p in model.parameters()):,}')

# ----------------------------------------------------------------------
# Load held-out v3 test dataset
# ----------------------------------------------------------------------
print(f'\n[2] Loading test data from {DATA_PATH} (HDF5)...')
di = H5DataImporter(
    path=DATA_PATH,
    data_types=['gex'],
    log_transform=False,
    top_percentile=100,
    min_features=100,
)
_, test_ds = di.import_data()
test_X = test_ds[:][0]['gex'].to(device)
test_y = test_ds[:][1]['uberon_tissue'].numpy()
print(f'    Test samples: {len(test_y)}')

label_mapping = model.dataset.label_mappings['uberon_tissue']
name_to_idx = {v: k for k, v in label_mapping.items()}
nan_idx = name_to_idx.get('nan', None)

# ----------------------------------------------------------------------
# Classifier-head predictions — unchanged from the original script
# ----------------------------------------------------------------------
print('\n[3] Running forward pass (classifier head)...')
with torch.no_grad():
    h = model.encoders[0](test_X)
    mu = model.FC_mean(h[0])
    logits = model.MLPs['uberon_tissue'](mu)
    if nan_idx is not None:
        logits[:, nan_idx] = -1e9
    pred_idx = logits.argmax(dim=1).cpu().numpy()

mask = ~np.isnan(test_y)
test_y_clean = test_y[mask].astype(int)
pred_clean = pred_idx[mask]
print(f'    Labelled samples: {mask.sum()}/{len(test_y)}')

ba = balanced_accuracy_score(test_y_clean, pred_clean)
f1 = f1_score(test_y_clean, pred_clean, average='weighted')
print(f'\nBalanced accuracy: {ba*100:.1f}%')
print(f'Weighted F1:       {f1*100:.1f}%')
print(f'Evaluated classes: {len(set(test_y_clean))}')

# ----------------------------------------------------------------------
# Per-class recall (tissue classification accuracy)
# ----------------------------------------------------------------------
classes_int = sorted(set(test_y_clean))
per_class_acc, per_class_n = {}, {}
for ci in classes_int:
    cn = label_mapping[ci]
    mask_cls = test_y_clean == ci
    per_class_acc[cn] = float((pred_clean[mask_cls] == ci).mean())
    per_class_n[cn] = int(mask_cls.sum())

# Retained for compatibility with the original plot-generation protocol.
if 'other' in per_class_acc:
    other_n = per_class_n.pop('other')
    del per_class_acc['other']
    print(f'    Excluded display-only catch-all "other" (n={other_n})')

sorted_cls = sorted(per_class_acc, key=lambda c: per_class_acc[c], reverse=True)
accuracies = [per_class_acc[c] for c in sorted_cls]
print(f'Displayed classes: {len(sorted_cls)}')
for cls in sorted_cls:
    print(f'  {cls:<22} n={per_class_n[cls]:4d}  acc={per_class_acc[cls]:.3f}')

# ----------------------------------------------------------------------
# Figure 2 — final Nature Communications artwork style
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 15))
y = np.arange(len(sorted_cls))
accuracy_pct = np.array([per_class_acc[c] * 100 for c in sorted_cls])
sample_counts = np.array([per_class_n[c] for c in sorted_cls])

cmap = plt.get_cmap('viridis')
norm = LogNorm(vmin=max(1, int(sample_counts.min())),
               vmax=int(sample_counts.max()))
ax.barh(y, accuracy_pct, color=cmap(norm(sample_counts)), height=0.72)
ax.invert_yaxis()  # highest accuracy on top

ax.set_yticks(y)
ax.set_yticklabels(
    [f"{c.replace('_', ' ')} (n={per_class_n[c]})" for c in sorted_cls],
    fontsize=9,
)
for i, value in enumerate(accuracy_pct):
    # Labels shown for every tissue, including those at/near 100%.
    ax.text(value + 0.35, i, f'{value:.0f}', va='center', fontsize=8,
            color='#333333')

ax.axvline(ba * 100, color='#D62728', linestyle='--', linewidth=1.5)
ax.text(
    ba * 100, 1.015, f'Balanced accuracy {ba * 100:.1f}%',
    transform=ax.get_xaxis_transform(), ha='center',
    color='#D62728', fontsize=11, clip_on=False,
)
ax.set_xlim(70, 103)
ax.set_xticks([70, 80, 90, 100])
ax.set_xlabel('Per-tissue classification accuracy (%)', fontsize=11)
ax.grid(axis='x', alpha=0.2)
ax.set_axisbelow(True)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.035)
cbar.set_label('Test samples (log scale)')
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:,.0f}'))

plt.tight_layout()
plt.savefig(OUT_SVG, dpi=300, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\n[4] Saved: {OUT_SVG} / {OUT_PNG}')

# ----------------------------------------------------------------------
# Save per-class table and classification metrics; schema retained
# ----------------------------------------------------------------------
pd.DataFrame({
    'tissue': sorted_cls,
    'accuracy': accuracies,
    'n_test': [per_class_n[c] for c in sorted_cls],
}).to_csv(OUT_CSV, index=False)
print(f'    Saved: {OUT_CSV}')

metrics = {
    'version': 'v3_118k_ScopeB_HDF5',
    'model': MODEL_PATH,
    'checkpoint': os.path.basename(best_ckpt),
    'n_test_total': int(len(test_y)),
    'n_test_labelled': int(mask.sum()),
    'balanced_accuracy': float(ba),
    'weighted_f1': float(f1),
    'n_classes': int(len(set(test_y_clean))),
    'per_class_accuracy': {c: float(per_class_acc[c]) for c in sorted_cls},
    'per_class_n': {c: int(per_class_n[c]) for c in sorted_cls},
}
with open(OUT_JSON, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'    Saved: {OUT_JSON}')
print('\n=== Done: Figure 2 final-format output ===')
