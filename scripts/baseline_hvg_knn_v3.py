#!/usr/bin/env python3
"""
v3 Baseline Benchmarks (118k Scope B HDF5)
============================================
Re-runs the v2 baseline kNN comparisons on the v3 tissue-curated training
compendium (118,263 train / 28,274 test) for apples-to-apples comparison
with the supervised VAE (94.9% balanced accuracy).

Baselines:
  1. All genes + kNN (k=5)
  2. Top-2000 HVG + kNN (k=5)
  3. Top-2000 HVG + PCA(50) + kNN (k=5)
  4. Top-2000 HVG + UMAP(50) + kNN (k=5)
  5. All genes + PCA(121) + kNN (k=5)    [matching VAE latent dim]

Adapted from baseline_hvg_knn.py — uses HDF5 directly (no CSV transpose).

Author: Amit Pande, MDC Berlin/BIMSB
"""
import h5py
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_DIR     = "processed_scaled_411k_tissue_B_h5"
OUT_FILE     = "baseline_results_v3.csv"
SUMMARY_FILE = "baseline_summary_v3.txt"
N_HVG        = 2000
K            = 5
PCA_LATENT   = 121  # match VAE latent dim
LABEL_COL    = 'uberon_tissue'

print("=" * 65)
print("v3 BASELINE BENCHMARKS — 118k Scope B HDF5")
print("=" * 65)

# ----------------------------------------------------------------------
# 1. Load HDF5 data
# ----------------------------------------------------------------------
print("\n[1] Loading HDF5 data...")
t0 = time.time()

def load_h5(path):
    with h5py.File(path, 'r') as f:
        X = f['expression'][:]                       # (samples, genes)
        sample_ids = [s.decode() for s in f['sample_ids'][:]]
        gene_symbols = [g.decode() for g in f['gene_symbols'][:]]
    return X, sample_ids, gene_symbols

train_X, train_ids, gene_symbols = load_h5(f"{DATA_DIR}/train/gex.h5")
test_X,  test_ids,  _            = load_h5(f"{DATA_DIR}/test/gex.h5")
print(f"   Train: {train_X.shape}, Test: {test_X.shape}")

train_clin = pd.read_csv(f"{DATA_DIR}/train/clin.csv", index_col=0)
test_clin  = pd.read_csv(f"{DATA_DIR}/test/clin.csv",  index_col=0)
print(f"   Clinical: train={len(train_clin)}, test={len(test_clin)}")
print(f"   Loaded in {time.time()-t0:.1f}s")

# ----------------------------------------------------------------------
# 2. Align samples with labels
# ----------------------------------------------------------------------
print(f"\n[2] Extracting labels ({LABEL_COL})...")
train_clin_aligned = train_clin.reindex(train_ids)
test_clin_aligned  = test_clin.reindex(test_ids)

train_y = train_clin_aligned[LABEL_COL].values
test_y  = test_clin_aligned[LABEL_COL].values

# Filter out unlabelled samples
def filter_labelled(X, y):
    mask = pd.notna(y) & (y != 'nan') & (y != 'other')
    return X[mask], y[mask], mask

train_X, train_y, train_mask = filter_labelled(train_X, train_y)
test_X,  test_y,  test_mask  = filter_labelled(test_X, test_y)
print(f"   After label filter: Train={len(train_X)}, Test={len(test_X)}")
print(f"   Unique classes: train={len(set(train_y))}, test={len(set(test_y))}")

# ----------------------------------------------------------------------
# 3. Handle NaN values in expression matrix (gene-mean imputation)
# ----------------------------------------------------------------------
n_nan_train = np.isnan(train_X).sum()
n_nan_test  = np.isnan(test_X).sum()
print(f"\n[3] NaN counts: train={n_nan_train}, test={n_nan_test}")
if n_nan_train > 0 or n_nan_test > 0:
    col_means = np.nanmean(train_X, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    for j in range(train_X.shape[1]):
        train_X[np.isnan(train_X[:, j]), j] = col_means[j]
        test_X[np.isnan(test_X[:, j]),  j] = col_means[j]
    print(f"   Imputed with gene means")

# ----------------------------------------------------------------------
# 4. Select Top-N highly variable genes (HVGs)
# ----------------------------------------------------------------------
print(f"\n[4] Selecting top {N_HVG} HVGs...")
gene_var = np.var(train_X, axis=0)
hvg_idx  = np.argsort(gene_var)[-N_HVG:]
train_X_hvg = train_X[:, hvg_idx]
test_X_hvg  = test_X[:,  hvg_idx]
print(f"   HVG variance range: {gene_var[hvg_idx].min():.4f}–{gene_var[hvg_idx].max():.4f}")

# ----------------------------------------------------------------------
# 5. Run baselines
# ----------------------------------------------------------------------
results = {}

def run_baseline(name, X_tr, X_te, y_tr, y_te, k=K):
    print(f"\n   → {name}...")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)
    ba  = balanced_accuracy_score(y_te, y_pred)
    f1w = f1_score(y_te, y_pred, average='weighted')
    elapsed = time.time() - t0
    print(f"      BA={ba:.4f}  F1={f1w:.4f}  ({elapsed:.1f}s)")
    results[name] = {'balanced_accuracy': float(ba),
                     'weighted_f1': float(f1w),
                     'time_s': float(elapsed)}

# Baseline 1: All genes
print("\n[5] Running baselines...")
run_baseline("All genes + kNN", train_X, test_X, train_y, test_y)

# Baseline 2: HVG only
run_baseline(f"Top-{N_HVG} HVG + kNN", train_X_hvg, test_X_hvg, train_y, test_y)

# Baseline 3: HVG + PCA(50)
print(f"\n   Computing PCA(50) on HVGs...")
pca50 = PCA(n_components=50, random_state=42)
train_pca50 = pca50.fit_transform(train_X_hvg)
test_pca50  = pca50.transform(test_X_hvg)
print(f"   Variance explained: {pca50.explained_variance_ratio_.sum():.3f}")
run_baseline(f"Top-{N_HVG} HVG + PCA(50) + kNN", train_pca50, test_pca50, train_y, test_y)

# Baseline 4: HVG + UMAP(50)
try:
    import umap
    print(f"\n   Computing UMAP(50) on HVGs...")
    t0 = time.time()
    reducer = umap.UMAP(n_components=50, random_state=42,
                        n_neighbors=15, min_dist=0.1, verbose=False)
    train_umap = reducer.fit_transform(train_X_hvg)
    test_umap  = reducer.transform(test_X_hvg)
    print(f"   UMAP done in {time.time()-t0:.1f}s")
    run_baseline(f"Top-{N_HVG} HVG + UMAP(50) + kNN",
                 train_umap, test_umap, train_y, test_y)
except ImportError:
    print("   UMAP not installed; skipping. Run: pip install umap-learn")

# Baseline 5: All genes + PCA(121)  [matches VAE latent dim]
print(f"\n   Computing PCA({PCA_LATENT}) on all genes...")
pca_full = PCA(n_components=PCA_LATENT, random_state=42)
train_pcaF = pca_full.fit_transform(train_X)
test_pcaF  = pca_full.transform(test_X)
print(f"   Variance explained: {pca_full.explained_variance_ratio_.sum():.3f}")
run_baseline(f"All genes + PCA({PCA_LATENT}) + kNN",
             train_pcaF, test_pcaF, train_y, test_y)

# Reference: VAE result from earlier eval
results["Supervised VAE (121-dim) [v3]"] = {
    'balanced_accuracy': 0.9488,
    'weighted_f1':       0.9619,
    'time_s':            None,
}

# ----------------------------------------------------------------------
# 6. Save + print summary
# ----------------------------------------------------------------------
print("\n" + "=" * 65)
print("v3 BASELINE RESULTS SUMMARY (118k Scope B)")
print("=" * 65)

df = pd.DataFrame(results).T.sort_values('balanced_accuracy', ascending=False)
df.index.name = 'Method'
print(df.to_string())
df.to_csv(OUT_FILE)

with open(SUMMARY_FILE, 'w') as f:
    f.write("v3 Baseline Benchmark Results (118k Scope B HDF5)\n")
    f.write("=" * 65 + "\n\n")
    for name, r in sorted(results.items(),
                          key=lambda x: x[1]['balanced_accuracy'], reverse=True):
        t = f" ({r['time_s']:.1f}s)" if r['time_s'] else ""
        f.write(f"{name:45s}  BA={r['balanced_accuracy']:.4f}"
                f"  F1={r['weighted_f1']:.4f}{t}\n")
    f.write("\nv2 (75k) reference baselines for comparison:\n")
    f.write("  All genes + kNN              BA=0.8962  F1=0.9172\n")
    f.write("  Top-2000 HVG + kNN           BA=0.8901  F1=0.9133\n")
    f.write("  Top-2000 HVG + PCA(50) + kNN BA=0.8460  F1=0.8957\n")
    f.write("  Top-2000 HVG + UMAP(50)+kNN  BA=0.8305  F1=0.8650\n")
    f.write("  All genes + PCA(121) + kNN   BA=0.8736  F1=0.9100\n")
    f.write("  Supervised VAE (v2 75k)      BA=0.9070  F1=0.9370\n")

print(f"\nSaved: {OUT_FILE}, {SUMMARY_FILE}")

# Comparison to v2
print("\n--- v2 vs v3 comparison ---")
v2_baselines = {
    'All genes + kNN':                       0.8962,
    f'Top-{N_HVG} HVG + kNN':                0.8901,
    f'Top-{N_HVG} HVG + PCA(50) + kNN':      0.8460,
    f'Top-{N_HVG} HVG + UMAP(50) + kNN':     0.8305,
    f'All genes + PCA({PCA_LATENT}) + kNN':  0.8736,
    'Supervised VAE (121-dim) [v3]':         0.9070,  # v2 VAE for reference
}
for name in v2_baselines:
    if name in results:
        v3 = results[name]['balanced_accuracy']
        v2 = v2_baselines[name]
        delta = (v3 - v2) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {name:40s}  v2={v2:.3f} → v3={v3:.3f}  ({sign}{delta:.1f} pp)")
