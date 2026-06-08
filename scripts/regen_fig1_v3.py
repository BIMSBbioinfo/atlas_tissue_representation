#!/usr/bin/env python3
"""
Figure 1 — v3 (118k Scope B HDF5)
==================================
t-SNE of latent embeddings from v3 Standard VAE (best checkpoint) on the
held-out test set (n=28,274). Kobak-Berens protocol: PCA init, multi-scale
perplexity, learning rate n/12. Also computes LISI + kNN source mixing.

Adapted from regen_fig1.py (v2). Uses HDF5 data + v3 model paths.
"""
import glob, os, sys, gc
import torch, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from openTSNE import TSNE as openTSNE
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, '.')
from h5_dataloader import H5DataImporter
from train_denoising_vae import DenoisingVAE  # registers the class

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RESULTS_DIR = 'results_denoising_vae_411k_B'
DATA_PATH   = 'processed_scaled_411k_tissue_B_h5'
STD_PTH     = f'{RESULTS_DIR}/standard_vae.pth'
STD_CKPT    = f'{RESULTS_DIR}/ckpt_standard/best_epoch=*.ckpt'

OUT_SVG     = 'Fig1_tsne_latent_space_v3_118k.svg'
OUT_PNG     = 'Fig1_tsne_latent_space_v3_118k.png'
COORDS_CSV  = 'tsne_kobak_coordinates_v3_118k.csv'

# GPU for encoding (faster on 28k samples × 105M params), CPU for t-SNE
device_encode = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------
# Load v3 model (CPU first → GPU for encoding)
# ----------------------------------------------------------------------
print(f"[1] Loading v3 Standard VAE...")
model = torch.load(STD_PTH, map_location='cpu', weights_only=False)
model = model.to(device_encode)
ckpts = sorted(glob.glob(STD_CKPT),
               key=lambda p: float(p.split('val_loss=')[1].split('.ckpt')[0]))
best_ckpt = ckpts[0]
print(f"    Best checkpoint: {os.path.basename(best_ckpt)}")
ckpt = torch.load(best_ckpt, map_location=device_encode, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"    Device: {device_encode}")

# ----------------------------------------------------------------------
# Load v3 test data via HDF5
# ----------------------------------------------------------------------
print(f"\n[2] Loading test data from {DATA_PATH}...")
di = H5DataImporter(path=DATA_PATH, data_types=['gex'],
                    log_transform=False, top_percentile=100, min_features=100)
_, test_ds = di.import_data()
print(f"    Test samples: {len(test_ds)}")

# ----------------------------------------------------------------------
# Extract latent embeddings (mu) via GPU
# ----------------------------------------------------------------------
print(f"\n[3] Extracting latent embeddings (GPU encoding)...")
loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
all_mu, all_samples = [], []
with torch.no_grad():
    for i, batch in enumerate(loader):
        dat, _, samples = batch
        x  = dat['gex'].to(device_encode)
        h  = model.encoders[0](x)
        mu = model.FC_mean(h[0])
        all_mu.append(mu.cpu().numpy())
        all_samples.extend(samples)
        if i % 20 == 0:
            print(f"    batch {i}/{len(loader)}", flush=True)

embeddings = np.concatenate(all_mu)
print(f"    Embeddings shape: {embeddings.shape}")

# Free GPU memory after encoding (t-SNE runs on CPU)
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------
# Load clinical metadata
# ----------------------------------------------------------------------
clin = pd.read_csv(f'{DATA_PATH}/test/clin.csv', index_col=0)
clin = clin.loc[all_samples]
print(f"    Sources: {clin['source'].value_counts().to_dict()}")

# ----------------------------------------------------------------------
# Kobak-Berens t-SNE
# ----------------------------------------------------------------------
n      = len(embeddings)
lr     = n / 12
perp_s = 30
perp_l = int(np.sqrt(n))
print(f"\n[4] Kobak-Berens t-SNE: n={n}, LR={lr:.0f}, perplexity={perp_s}")

tsne = openTSNE(
    n_components=2,
    perplexity=perp_s,
    learning_rate=lr,
    n_iter=750,
    initialization='pca',
    metric='cosine',
    n_jobs=4,
    random_state=42,
    verbose=True,
)
print(f"    Running t-SNE on {n} samples (~20-40 min for 28k)...")
coords = tsne.fit(embeddings)
print(f"    t-SNE done. Shape: {coords.shape}")

# Save coordinates for downstream use
coords_df = pd.DataFrame(coords, columns=['tSNE1', 'tSNE2'], index=all_samples)
coords_df['source']        = clin['source'].values
coords_df['uberon_tissue'] = clin['uberon_tissue'].values
coords_df.to_csv(COORDS_CSV)
print(f"    Saved: {COORDS_CSV}")

# ----------------------------------------------------------------------
# Batch mixing metrics
# ----------------------------------------------------------------------
print(f"\n[5] Computing batch mixing metrics...")
sources = clin['source'].values
unique_sources = sorted(set(sources))
print(f"    Unique sources: {unique_sources}")

k = 20
nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(embeddings)
_, indices = nn.kneighbors(embeddings)

knn_mixing = []
for i in range(len(embeddings)):
    nbrs = indices[i, 1:]   # exclude self
    own  = sources[i]
    diff = np.sum(sources[nbrs] != own) / k
    knn_mixing.append(diff)
knn_mix = float(np.mean(knn_mixing))
print(f"    kNN source mixing (k=20): {knn_mix:.4f}")

# LISI
def compute_lisi(emb, labels, k=20):
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(emb)
    _, indices = nn.kneighbors(emb)
    lisi_scores = []
    unique_labels = list(set(labels))
    for i in range(len(emb)):
        nbrs = labels[indices[i, 1:]]
        counts = {l: np.sum(nbrs == l) for l in unique_labels}
        p_sq = sum((c/k)**2 for c in counts.values() if c > 0)
        lisi_scores.append(1.0 / p_sq if p_sq > 0 else 1.0)
    return float(np.mean(lisi_scores))

lisi = compute_lisi(embeddings, sources, k=20)
n_sources = len(unique_sources)
print(f"    LISI (source): {lisi:.4f} (max possible: {n_sources}.0)")

print(f"\n{'='*55}")
print(f"v3 BATCH MIXING METRICS")
print(f"  kNN source mixing (k=20): {knn_mix:.3f}")
print(f"  LISI (source):            {lisi:.2f} / {n_sources}.0")
print(f"  t-SNE: n={n}, LR={lr:.0f}, perplexity {perp_s}")
print(f"{'='*55}")
print(f"\nv2 reference: kNN={0.047}, LISI={1.09}/4.0")

# ----------------------------------------------------------------------
# Organ system / source colour mapping
# ----------------------------------------------------------------------
organ_map = {
    'brain':'Brain', 'spinal_cord':'Brain', 'nerve':'Brain', 'pituitary':'Brain',
    'blood':'Blood/Immune', 'blood_vessel':'Blood/Immune', 'bone_marrow':'Blood/Immune',
    'lymphoid':'Blood/Immune', 'spleen':'Blood/Immune', 'thymus':'Blood/Immune',
    'lung':'Thoracic', 'pleura':'Thoracic', 'heart':'Thoracic',
    'liver':'GI tract', 'colon':'GI tract', 'stomach':'GI tract',
    'pancreas':'GI tract', 'esophagus':'GI tract', 'small_intestine':'GI tract',
    'biliary_tract':'GI tract',
    'kidney':'Urinary', 'bladder':'Urinary',
    'breast':'Reproductive', 'ovary':'Reproductive', 'uterus':'Reproductive',
    'cervix':'Reproductive', 'vagina':'Reproductive', 'testis':'Reproductive',
    'prostate':'Reproductive', 'placenta':'Reproductive',
    'skin':'Skin/Connective', 'fibroblast':'Skin/Connective',
    'soft_tissue':'Skin/Connective', 'bone':'Skin/Connective', 'muscle':'Skin/Connective',
    'adipose':'Endocrine', 'adrenal_gland':'Endocrine', 'thyroid':'Endocrine',
    'salivary_gland':'Endocrine',
    'head_and_neck':'Head & Neck',
    'eye':'Other', 'stem_cell':'Other', 'other':'Other',
}
organ_colors = {
    'Brain':          '#4472C4',
    'Blood/Immune':   '#C0504D',
    'Thoracic':       '#9BBB59',
    'GI tract':       '#8064A2',
    'Urinary':        '#00B0F0',
    'Reproductive':   '#FF8C00',
    'Skin/Connective':'#7F4F24',
    'Endocrine':      '#FF69B4',
    'Head & Neck':    '#2E8B57',
    'Other':          '#AAAAAA',
}
source_colors = {
    'TCGA':   '#E74C3C',
    'GTEx':   '#3498DB',
    'DepMap': '#27AE60',
    'ARCHS4': '#E6A817',
}

tissues  = clin['uberon_tissue'].fillna('other').values
organs   = np.array([organ_map.get(t, 'Other') for t in tissues])
sources_arr = clin['source'].values

# ----------------------------------------------------------------------
# Plot — same style as v2 Fig 1
# ----------------------------------------------------------------------
print(f"\n[6] Plotting Fig 1...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Panel a — by organ system
for organ in organ_colors:
    mask = organs == organ
    if mask.sum() == 0:
        continue
    ax1.scatter(coords[mask, 0], coords[mask, 1],
                c=organ_colors[organ], s=2, alpha=0.5, linewidths=0,
                rasterized=True)
    # Label at centroid
    cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
    ax1.text(cx, cy, organ, fontsize=7, ha='center', va='center',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor=organ_colors[organ], alpha=0.75, linewidth=0.8))

ax1.set_xlabel('t-SNE 1', fontsize=12)
ax1.set_ylabel('t-SNE 2', fontsize=12)
ax1.set_title('a', fontsize=18, fontweight='bold', loc='left')
ax1.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
for sp in ax1.spines.values():
    sp.set_linewidth(0.5); sp.set_color('#CCCCCC')

# Panel b — by source
for src in ['GTEx', 'TCGA', 'DepMap', 'ARCHS4']:
    mask = sources_arr == src
    if mask.sum() == 0:
        continue
    ax2.scatter(coords[mask, 0], coords[mask, 1],
                c=source_colors[src], s=2, alpha=0.5, linewidths=0,
                rasterized=True, label=src)

handles = [Line2D([0],[0], marker='o', color='w',
                  markerfacecolor=source_colors[s], markersize=10,
                  markeredgewidth=0.8, markeredgecolor='white', label=s)
           for s in ['DepMap', 'ARCHS4', 'TCGA', 'GTEx']
           if s in unique_sources]
ax2.legend(handles=handles, fontsize=11, frameon=True, loc='lower right',
           facecolor='white', edgecolor='#CCCCCC', framealpha=0.95)
ax2.set_xlabel('t-SNE 1', fontsize=12)
ax2.set_ylabel('t-SNE 2', fontsize=12)
ax2.set_title('b', fontsize=18, fontweight='bold', loc='left')
ax2.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
for sp in ax2.spines.values():
    sp.set_linewidth(0.5); sp.set_color('#CCCCCC')

plt.tight_layout(w_pad=3)
plt.savefig(OUT_SVG, dpi=300, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
print(f"    Saved: {OUT_SVG} / {OUT_PNG}")
