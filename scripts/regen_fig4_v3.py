#!/usr/bin/env python3
"""
TARGET cross-validation — v3 (118k Scope B model)
==================================================

Evaluates how well the v3 Standard VAE generalises to TARGET paediatric
cancers (734 samples, 7 cancer types) — independent dataset never seen
during training.

Protocol (matches v2 manuscript Methods):
  1. Load v3 Standard VAE (best checkpoint)
  2. Load TARGET test data, align to 16,115 model genes
  3. Per-gene z-score normalisation (heuristic cross-platform alignment)
  4. Encode TARGET samples → 121-dim latent
  5. Encode v3 training-set test split → reference latent
  6. kNN (k=5, Euclidean) classifier in latent space
  7. Per-cancer-type breakdown + manuscript-style figure

Outputs:
  - results_denoising_vae_411k_B/target_v3_results.json
  - Fig4_TARGET_validation_v3_118k.svg / .png
"""
import json
import glob
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, '.')
from h5_dataloader import H5DataImporter

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RESULTS_DIR  = 'results_denoising_vae_411k_B'
MODEL_PATH   = f'{RESULTS_DIR}/standard_vae.pth'
CKPT_GLOB    = f'{RESULTS_DIR}/ckpt_standard/best_epoch=*.ckpt'
TRAIN_DATA   = 'processed_scaled_411k_tissue_B_h5'
TARGET_GEX   = 'held_out/target_flexynesis/test/gex.csv'
TARGET_CLIN  = 'held_out/target_flexynesis/test/clin.csv'

OUT_JSON     = f'{RESULTS_DIR}/target_v3_results.json'
OUT_SVG      = 'Fig4_TARGET_validation_v3_118k.svg'
OUT_PNG      = 'Fig4_TARGET_validation_v3_118k.png'

K_NEIGHBOURS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------
# Step 1: Load v3 model
# ----------------------------------------------------------------------
print(f"[1] Loading v3 Standard VAE...")
model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = model.to(device)
best_ckpt = sorted(glob.glob(CKPT_GLOB),
                   key=lambda p: float(p.split('val_loss=')[1].split('.ckpt')[0]))[0]
print(f"    Best checkpoint: {os.path.basename(best_ckpt)}")
ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

label_mapping = model.dataset.label_mappings['uberon_tissue']
idx_to_name   = {k: v for k, v in label_mapping.items()}
name_to_idx   = {v: k for k, v in label_mapping.items()}

# ----------------------------------------------------------------------
# Step 2: Encode v3 training-set test split as REFERENCE latent space
#         (this is the corpus the kNN classifier searches against)
# ----------------------------------------------------------------------
print(f"\n[2] Loading v3 test set (reference for kNN)...")
di = H5DataImporter(path=TRAIN_DATA, data_types=['gex'],
                    log_transform=False, top_percentile=100, min_features=100)
_, ref_ds = di.import_data()
ref_X = ref_ds[:][0]['gex'].to(device)
ref_y = ref_ds[:][1]['uberon_tissue'].numpy()

# Get the gene order the model expects
model_genes = ref_ds.features['gex']
print(f"    Model genes: {len(model_genes)}")
print(f"    Reference samples: {len(ref_y)}")

# Encode reference set → latent
print(f"\n[3] Encoding reference set → latent space...")
with torch.no_grad():
    h_ref = model.encoders[0](ref_X)
    ref_latent = model.FC_mean(h_ref[0]).cpu().numpy()
print(f"    Reference latent shape: {ref_latent.shape}")

# Filter labelled-only (drop NaN tissue labels)
nan_idx = name_to_idx.get('nan', None)
ref_mask = ~np.isnan(ref_y)
if nan_idx is not None:
    ref_mask = ref_mask & (ref_y != nan_idx)
ref_latent_labelled = ref_latent[ref_mask]
ref_y_labelled = ref_y[ref_mask].astype(int)
print(f"    Labelled reference samples: {len(ref_y_labelled)}")

# ----------------------------------------------------------------------
# Step 3: Load TARGET data + align genes + normalise
# ----------------------------------------------------------------------
print(f"\n[4] Loading TARGET data...")
target_gex = pd.read_csv(TARGET_GEX, index_col=0)  # genes × samples
target_clin = pd.read_csv(TARGET_CLIN, index_col=0)
print(f"    TARGET raw shape: {target_gex.shape}")
print(f"    TARGET clinical samples: {len(target_clin)}")

# Drop Entrez_Gene_Id if present (some held-out files have it)
for col in ['Entrez_Gene_Id', 'entrez_id']:
    if col in target_gex.columns:
        target_gex = target_gex.drop(columns=[col])
        print(f"    Dropped column: {col}")

# Align genes (16,115 model genes; fill missing with 0)
print(f"\n[5] Aligning to model gene set...")
target_genes = target_gex.index.tolist()
common_genes = [g for g in model_genes if g in target_genes]
missing_genes = [g for g in model_genes if g not in target_genes]
print(f"    Common genes: {len(common_genes)}/{len(model_genes)}")
print(f"    Missing (filled with 0): {len(missing_genes)}")

# Build aligned matrix: rows = model_genes order, cols = TARGET samples
aligned = pd.DataFrame(0.0, index=model_genes, columns=target_gex.columns)
aligned.loc[common_genes] = target_gex.loc[common_genes].values.astype(np.float32)

# Per-gene z-score normalisation (cross-platform alignment heuristic)
# Match training distribution: zero-mean unit-variance per gene
target_X = aligned.values.T.astype(np.float32)  # samples × genes
print(f"    TARGET X shape (samples × genes): {target_X.shape}")

# Z-score per gene (over TARGET samples)
target_X = (target_X - target_X.mean(axis=0, keepdims=True)) / (
    target_X.std(axis=0, keepdims=True) + 1e-8)
target_X = np.nan_to_num(target_X, nan=0.0, posinf=0.0, neginf=0.0)

# Also rescale to roughly match training feature distribution
# Training reference: use ref_latent's input distribution stats
ref_X_np = ref_X.cpu().numpy()
ref_mean = ref_X_np.mean(axis=0)
ref_std  = ref_X_np.std(axis=0) + 1e-8
target_X = target_X * ref_std + ref_mean  # rescale to training distribution

# Encode TARGET → latent
print(f"\n[6] Encoding TARGET → latent space...")
target_X_torch = torch.FloatTensor(target_X).to(device)
with torch.no_grad():
    h_tgt = model.encoders[0](target_X_torch)
    target_latent = model.FC_mean(h_tgt[0]).cpu().numpy()
print(f"    TARGET latent shape: {target_latent.shape}")

# ----------------------------------------------------------------------
# Step 4: kNN classification in latent space
# ----------------------------------------------------------------------
print(f"\n[7] kNN classification (k={K_NEIGHBOURS}, Euclidean)...")
knn = KNeighborsClassifier(n_neighbors=K_NEIGHBOURS, metric='euclidean', n_jobs=-1)
knn.fit(ref_latent_labelled, ref_y_labelled)
target_pred_idx = knn.predict(target_latent)
target_pred_names = [idx_to_name[i] for i in target_pred_idx]

# ----------------------------------------------------------------------
# Step 5: Match clinical labels + per-cancer breakdown
# ----------------------------------------------------------------------
print(f"\n[8] Computing per-cancer accuracy...")

# Align clinical rows with target_gex columns (sample IDs)
target_clin_aligned = target_clin.reindex(target_gex.columns)
cancer_types = target_clin_aligned['tissue_type'].values

# Define expected tissue mappings per cancer (manuscript developmental biology)
EXPECTED = {
    'Acute Myeloid Leukemia':                         {'blood', 'bone_marrow', 'lymphoid'},
    'Acute Lymphoblastic Leukemia':                   {'blood', 'bone_marrow', 'lymphoid'},
    'Acute Myeloid Leukemia, Induction Failure Subproject': {'blood', 'bone_marrow', 'lymphoid'},
    'Neuroblastoma':                                  {'brain', 'nerve', 'adrenal_gland'},
    'Wilms Tumor':                                    {'kidney', 'ovary', 'uterus', 'testis'},
    'Kidney, Rhabdoid Tumor':                         {'kidney'},
    'Clear Cell Sarcoma Of The Kidney':               {'kidney'},
}

# Per-cancer aggregation
per_cancer = {}
for ct in sorted(set(cancer_types)):
    mask = cancer_types == ct
    n = int(mask.sum())
    if n == 0:
        continue
    preds = [target_pred_names[i] for i in range(len(target_pred_names)) if mask[i]]
    pred_counts = dict(Counter(preds).most_common())
    expected = EXPECTED.get(ct, set())
    n_correct = sum(1 for p in preds if p in expected)
    per_cancer[ct] = {
        'n': n,
        'correct': n_correct,
        'accuracy': n_correct / n,
        'expected_tissues': sorted(expected),
        'predictions': pred_counts,
    }

overall_n = sum(c['n'] for c in per_cancer.values())
overall_correct = sum(c['correct'] for c in per_cancer.values())
overall_acc = overall_correct / overall_n if overall_n > 0 else 0.0

# ----------------------------------------------------------------------
# Print summary
# ----------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"v3 TARGET CROSS-VALIDATION RESULTS (118k Scope B model)")
print(f"{'='*70}")
print(f"Overall: {overall_correct}/{overall_n} = {overall_acc*100:.1f}%")
print()
for ct in sorted(per_cancer, key=lambda x: -per_cancer[x]['n']):
    c = per_cancer[ct]
    print(f"  {ct[:50]:<52} n={c['n']:4d}  acc={c['accuracy']*100:5.1f}%")
    top3 = list(c['predictions'].items())[:3]
    pred_str = ', '.join(f"{k}={v}" for k, v in top3)
    print(f"      Top predictions: {pred_str}")
print(f"{'='*70}")
print(f"\nv2 baseline (75k_v2, manuscript): 86.6% overall")
print(f"v3 result:                        {overall_acc*100:.1f}% overall")

# ----------------------------------------------------------------------
# Save JSON
# ----------------------------------------------------------------------
results = {
    'version': 'v3_118k_ScopeB_HDF5',
    'model': MODEL_PATH,
    'checkpoint': os.path.basename(best_ckpt),
    'n_samples': overall_n,
    'overall_correct': overall_correct,
    'overall_accuracy': overall_acc,
    'k_neighbours': K_NEIGHBOURS,
    'per_cancer': per_cancer,
    'baseline_v2_manuscript': {'overall_accuracy': 0.866},
}
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {OUT_JSON}")

# ----------------------------------------------------------------------
# Plot — manuscript-style stacked bars
# ----------------------------------------------------------------------
print(f"\n[9] Generating Fig 4...")

# Colour scheme matching v2 manuscript
COLOURS = {
    'blood':         '#C0392B',  # red — haematopoietic
    'bone_marrow':   '#C0392B',
    'lymphoid':      '#C0392B',
    'brain':         '#2166AC',  # blue — neural
    'nerve':         '#2166AC',
    'adrenal_gland': '#4A90D9',  # lighter blue
    'kidney':        '#888888',  # grey — urogenital
    'ovary':         '#555555',  # darker grey
    'uterus':        '#777777',
    'testis':        '#999999',
}
UNEXPECTED = '#EEEEEE'

# Order cancers by n desc, then plot
order = [
    'Acute Myeloid Leukemia',
    'Acute Lymphoblastic Leukemia',
    'Acute Myeloid Leukemia, Induction Failure Subproject',
    'Neuroblastoma',
    'Wilms Tumor',
    'Kidney, Rhabdoid Tumor',
    'Clear Cell Sarcoma Of The Kidney',
]
order = [c for c in order if c in per_cancer]

# Display labels (short)
SHORT = {
    'Acute Myeloid Leukemia': 'Acute Myeloid Leukemia',
    'Acute Lymphoblastic Leukemia': 'Acute Lymphoblastic Leukemia',
    'Acute Myeloid Leukemia, Induction Failure Subproject': 'AML Induction Failure',
    'Neuroblastoma': 'Neuroblastoma',
    'Wilms Tumor': 'Wilms Tumor',
    'Kidney, Rhabdoid Tumor': 'Rhabdoid Tumor',
    'Clear Cell Sarcoma Of The Kidney': 'Clear Cell Sarcoma',
}

fig, ax = plt.subplots(figsize=(12, 8))
y_positions = list(range(len(order)))[::-1]

for y, ct in zip(y_positions, order):
    c = per_cancer[ct]
    expected = EXPECTED.get(ct, set())
    n = c['n']
    label = f"{SHORT.get(ct, ct)}\n(n={n})"

    # Build segments in descending order; group expected (with proper colour),
    # and lump unexpected into one grey block
    segments_expected = []
    unexpected_total = 0
    for tissue, count in c['predictions'].items():
        frac = count / n
        if tissue in expected:
            segments_expected.append((tissue, frac, COLOURS.get(tissue, '#888888')))
        else:
            unexpected_total += frac

    # Plot expected segments first
    x = 0
    for tissue, frac, colour in segments_expected:
        ax.barh(y, frac, left=x, height=0.62, color=colour,
                edgecolor='white', linewidth=0.8)
        if frac >= 0.07:
            ax.text(x + frac/2, y, tissue.replace('_', ' '),
                    ha='center', va='center', fontsize=8.5,
                    color='white', fontweight='bold')
        x += frac
    # Then unexpected as single grey block
    if unexpected_total > 0.001:
        ax.barh(y, unexpected_total, left=x, height=0.62, color=UNEXPECTED,
                edgecolor='#AAAAAA', linewidth=0.5)
        if unexpected_total >= 0.07:
            ax.text(x + unexpected_total/2, y, 'other',
                    ha='center', va='center', fontsize=8.5, color='#555555')

    # Percentage label on right
    pct = c['accuracy'] * 100
    bold = pct >= 80
    ax.text(1.02, y, f'{pct:.0f}%', va='center', fontsize=11,
            color='black' if bold else '#777777',
            fontweight='bold' if bold else 'normal')

ax.set_yticks(y_positions)
ax.set_yticklabels([f"{SHORT.get(ct, ct)}\n(n={per_cancer[ct]['n']})" for ct in order],
                   fontsize=10)
ax.set_xlabel('Fraction of samples', fontsize=12)
ax.set_xlim(0, 1.0)
ax.set_ylim(-1.2, len(order) - 0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_handles = [
    mpatches.Patch(color='#C0392B', label='Blood / Immune'),
    mpatches.Patch(color='#2166AC', label='Brain / CNS / Nerve'),
    mpatches.Patch(color='#4A90D9', label='Adrenal gland'),
    mpatches.Patch(color='#888888', label='Urinary / Reproductive'),
    mpatches.Patch(color=UNEXPECTED, label='Unexpected tissue',
                   edgecolor='#AAAAAA', linewidth=0.8),
]
ax.legend(handles=legend_handles, fontsize=9, loc='lower right',
          frameon=True, facecolor='white', edgecolor='#CCCCCC', framealpha=0.95)

ax.text(1.0, -1.05,
        f'Overall: {overall_acc*100:.1f}% ({overall_correct}/{overall_n})',
        ha='right', va='center', fontsize=9.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#AAAAAA', alpha=0.9))

plt.tight_layout()
plt.savefig(OUT_SVG, dpi=300, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
print(f"Saved: {OUT_SVG}")
print(f"Saved: {OUT_PNG}")
print(f"\n=== Done. v3 TARGET overall: {overall_acc*100:.1f}% ===")
