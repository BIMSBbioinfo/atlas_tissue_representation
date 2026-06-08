#!/usr/bin/env python3
"""
Figure 2 — v3 (118k Scope B HDF5).

Adapted from regen_fig2_final.py (v2, 75k_v2).

Only changes:
  - Model path     : results_denoising_vae_411k_B/standard_vae.pth
  - Checkpoint     : auto-detected from ckpt_standard/
  - Data path      : processed_scaled_411k_tissue_B_h5
  - DataImporter   : H5DataImporter (HDF5 path, not CSV)

Same evaluation protocol, same plot style, same output schema —
guarantees direct comparability with v2 Figure 2 in the manuscript.
"""
import glob
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, '.')
from h5_dataloader import H5DataImporter

# ----------------------------------------------------------------------
# CONFIG — v3 paths
# ----------------------------------------------------------------------
RESULTS_DIR = 'results_denoising_vae_411k_B'
MODEL_PATH  = f'{RESULTS_DIR}/standard_vae.pth'
CKPT_GLOB   = f'{RESULTS_DIR}/ckpt_standard/best_epoch=*.ckpt'
DATA_PATH   = 'processed_scaled_411k_tissue_B_h5'

OUT_SVG = 'Fig2_per_class_accuracy_v3_118k.svg'
OUT_PNG = 'Fig2_per_class_accuracy_v3_118k.png'
OUT_CSV = 'fig2_perclass_v3_118k.csv'
OUT_JSON = f'{RESULTS_DIR}/classification_metrics_v3.json'  # for manuscript table

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------
# Load model + best checkpoint
# ----------------------------------------------------------------------
print(f"[1] Loading Standard VAE from {MODEL_PATH}...")
model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = model.to(device)
    
model.to(device)

# Auto-detect best checkpoint (lowest val_loss in filename)
ckpts = glob.glob(CKPT_GLOB)
if not ckpts:
    sys.exit(f"ERROR: No checkpoints found at {CKPT_GLOB}")
# Best = lowest val_loss embedded in filename
best_ckpt = sorted(ckpts, key=lambda p: float(p.split('val_loss=')[1].split('.ckpt')[0]))[0]
print(f"    Best checkpoint: {os.path.basename(best_ckpt)}")

ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"    Model loaded, parameters: {sum(p.numel() for p in model.parameters()):,}")

# ----------------------------------------------------------------------
# Load v3 data via H5DataImporter (HDF5 path)
# ----------------------------------------------------------------------
print(f"\n[2] Loading test data from {DATA_PATH} (HDF5)...")
di = H5DataImporter(
    path=DATA_PATH,
    data_types=['gex'],
    log_transform=False,
    top_percentile=100,
    min_features=100,
)
_, test_ds = di.import_data()

# ----------------------------------------------------------------------
# Extract X, y from test dataset
# ----------------------------------------------------------------------
test_X = test_ds[:][0]['gex'].to(device)
test_y = test_ds[:][1]['uberon_tissue'].numpy()
print(f"    Test samples: {len(test_y)}")

# Label mappings
label_mapping = model.dataset.label_mappings['uberon_tissue']
name_to_idx   = {v: k for k, v in label_mapping.items()}
nan_idx       = name_to_idx.get('nan', None)

# ----------------------------------------------------------------------
# Forward pass — classify
# ----------------------------------------------------------------------
print(f"\n[3] Running forward pass (classifier head)...")
with torch.no_grad():
    h      = model.encoders[0](test_X)
    mu     = model.FC_mean(h[0])
    logits = model.MLPs['uberon_tissue'](mu)
    if nan_idx is not None:
        logits[:, nan_idx] = -1e9
    pred_idx = logits.argmax(dim=1).cpu().numpy()

# Exclude NaN (unlabelled) test samples — same as v2 protocol
mask         = ~np.isnan(test_y)
test_y_clean = test_y[mask].astype(int)
pred_clean   = pred_idx[mask]
print(f"    Labelled samples: {mask.sum()}/{len(test_y)}")

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
ba = balanced_accuracy_score(test_y_clean, pred_clean)
f1 = f1_score(test_y_clean, pred_clean, average='weighted')
print(f"\n{'='*60}")
print(f"v3 CLASSIFICATION RESULTS (118k Scope B HDF5)")
print(f"{'='*60}")
print(f"Balanced accuracy : {ba:.4f}  ({ba*100:.1f}%)")
print(f"Weighted F1       : {f1:.4f}  ({f1*100:.1f}%)")
print(f"Classes           : {len(set(test_y_clean))}")
print(f"{'='*60}")
print(f"\nv2 baseline (75k_v2): BA=90.7%, F1=93.7%")

# ----------------------------------------------------------------------
# Per-class accuracy
# ----------------------------------------------------------------------
classes_int = sorted(set(test_y_clean))
per_class_acc, per_class_n = {}, {}
for ci in classes_int:
    cn   = label_mapping[ci]
    mask_cls = test_y_clean == ci
    per_class_acc[cn] = float((pred_clean[mask_cls] == ci).mean())
    per_class_n[cn]   = int(mask_cls.sum())

# Remove 'other' catch-all if present (same as v2)
removed_other = False
if 'other' in per_class_acc:
    del per_class_acc['other']
    del per_class_n['other']
    removed_other = True
    print("\n(removed 'other' from display — n=" 
          + str(per_class_n.get('other', 'N/A')) + " catch-all category)")

sorted_cls = sorted(per_class_acc, key=lambda c: per_class_acc[c], reverse=True)
accuracies = [per_class_acc[c] for c in sorted_cls]
print(f"Displayed classes : {len(sorted_cls)}")

print("\nPer-class results (sorted by accuracy):")
for cls in sorted_cls:
    print(f"  {cls:<22} n={per_class_n[cls]:4d}  acc={per_class_acc[cls]:.3f}")

# ----------------------------------------------------------------------
# Plot — same style as v2 Fig 2
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 14))
y = np.arange(len(sorted_cls))
ax.barh(y, accuracies, color='#1a3d6e', height=0.7)
for i, cls in enumerate(sorted_cls):
    ax.text(1.01, i, f"n={per_class_n[cls]}", va='center', fontsize=7.5, color='gray')
    if per_class_acc[cls] >= 1.0:
        ax.text(per_class_acc[cls]-0.02, i, '100%', va='center', ha='right',
                fontsize=8, color='red', fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels([c.replace('_', ' ') for c in sorted_cls], fontsize=8.5)
ax.set_xlabel('Classification accuracy', fontsize=12)
ax.set_xlim(0, 1.15)
ax.axvline(x=ba, color='#888888', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(0.38, 3,
        f'v3 (118k):\nBalanced accuracy = {ba*100:.1f}%\nWeighted F1 = {f1*100:.1f}%',
        transform=ax.get_yaxis_transform(), fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='gray', alpha=0.9))
plt.tight_layout()
plt.savefig(OUT_SVG, dpi=300, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
print(f"\n[4] Saved: {OUT_SVG} / {OUT_PNG}")

# ----------------------------------------------------------------------
# Persist artifacts
# ----------------------------------------------------------------------
pd.DataFrame({
    'tissue':   sorted_cls,
    'accuracy': accuracies,
    'n_test':   [per_class_n[c] for c in sorted_cls],
}).to_csv(OUT_CSV, index=False)
print(f"    Saved: {OUT_CSV}")

# JSON output for manuscript Table 1 integration
import json
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
    'baseline_v2_75k': {'balanced_accuracy': 0.907, 'weighted_f1': 0.937},
}
with open(OUT_JSON, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"    Saved: {OUT_JSON}")
print(f"\n=== Done. v3 vs v2: BA {ba*100:.1f}% (v3) vs 90.7% (v2) ===")
