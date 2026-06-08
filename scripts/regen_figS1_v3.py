#!/usr/bin/env python3
"""FigS1: Confusion matrix — v3 (118k Scope B HDF5)"""
import glob, os, sys
import torch, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

sys.path.insert(0, '.')
from h5_dataloader import H5DataImporter

RESULTS_DIR = 'results_denoising_vae_411k_B'
MODEL_PATH  = f'{RESULTS_DIR}/standard_vae.pth'
CKPT_GLOB   = f'{RESULTS_DIR}/ckpt_standard/best_epoch=*.ckpt'
DATA_PATH   = 'processed_scaled_411k_tissue_B_h5'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (CPU first → GPU to avoid OOM)
print(f"[1] Loading Standard VAE...")
model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = model.to(device)

best_ckpt = sorted(glob.glob(CKPT_GLOB),
                   key=lambda p: float(p.split('val_loss=')[1].split('.ckpt')[0]))[0]
print(f"    Best checkpoint: {os.path.basename(best_ckpt)}")
ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

label_mapping = model.dataset.label_mappings['uberon_tissue']
name_to_idx   = {v: k for k, v in label_mapping.items()}
nan_idx       = name_to_idx.get('nan', None)
other_idx     = name_to_idx.get('other', None)

# Load v3 data via HDF5
print(f"\n[2] Loading test data (HDF5)...")
di = H5DataImporter(path=DATA_PATH, data_types=['gex'],
                    log_transform=False, top_percentile=100, min_features=100)
_, test_ds = di.import_data()
test_X = test_ds[:][0]['gex'].to(device)
test_y = test_ds[:][1]['uberon_tissue'].numpy()

# Forward pass
print(f"\n[3] Forward pass...")
with torch.no_grad():
    h = model.encoders[0](test_X)
    mu = model.FC_mean(h[0])
    logits = model.MLPs['uberon_tissue'](mu)
    if nan_idx is not None: logits[:, nan_idx] = -1e9
    pred_idx = logits.argmax(dim=1).cpu().numpy()

# Filter labelled, exclude 'other'
mask = ~np.isnan(test_y)
if other_idx is not None:
    mask = mask & (test_y != other_idx)
y_true = test_y[mask].astype(int)
y_pred = pred_idx[mask]

ba = balanced_accuracy_score(y_true, y_pred)
print(f"    Balanced accuracy: {ba*100:.1f}%")

# Get class names
classes = sorted(set(y_true))
class_names = [label_mapping[i].replace('_', ' ') for i in classes]
cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')

# Plot — same style as v2
fig, ax = plt.subplots(figsize=(14, 13))
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fraction')
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(class_names, rotation=90, fontsize=7)
ax.set_yticklabels(class_names, fontsize=7)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title(f'Tissue Classification Confusion Matrix (Normalised)\n'
             f'v3 model (118K), {ba*100:.1f}% balanced accuracy',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('FigS1_confusion_matrix_v3_118k.svg', dpi=300, bbox_inches='tight')
plt.savefig('FigS1_confusion_matrix_v3_118k.png', dpi=300, bbox_inches='tight')

# Also save CM data for downstream analysis
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
    'FigS1_confusion_matrix_v3_118k.csv')

print(f"\n[4] Saved: FigS1_confusion_matrix_v3_118k.{{svg,png,csv}}")
print(f"\nv2 baseline: 90.7% balanced accuracy")
print(f"v3 result:   {ba*100:.1f}% balanced accuracy")
