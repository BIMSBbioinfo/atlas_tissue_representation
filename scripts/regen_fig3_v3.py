#!/usr/bin/env python3
"""
Figure 3 — v3 (118k Scope B HDF5)
==================================
Reconstruction quality (per-gene + per-sample Spearman ρ) and gene imputation
for both Standard and Denoising VAEs trained on the v3 compendium.

CPU inference (both models 8GB, won't fit on RTX 4060). Saves per-gene and
per-sample arrays as .npy for Fig S2 reuse.

Adapted from regen_fig3.py (v2).
"""
import glob, os, sys, gc
import torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from h5_dataloader import H5DataImporter
from train_denoising_vae import DenoisingVAE  # for the class registration

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RESULTS_DIR = 'results_denoising_vae_411k_B'
DATA_PATH   = 'processed_scaled_411k_tissue_B_h5'

STD_PTH     = f'{RESULTS_DIR}/standard_vae.pth'
STD_CKPT    = f'{RESULTS_DIR}/ckpt_standard/best_epoch=*.ckpt'
DN_PTH      = f'{RESULTS_DIR}/denoising_vae.pth'
DN_CKPT     = f'{RESULTS_DIR}/ckpt_denoising/best_epoch=*.ckpt'

OUT_SVG     = 'Fig3_reconstruction_denoising_v3_118k.svg'
OUT_PNG     = 'Fig3_reconstruction_denoising_v3_118k.png'
ARRAYS_NPZ  = f'{RESULTS_DIR}/fig3_arrays_v3.npz'

# CPU inference — both models ~8GB each, won't fit on 8GB RTX 4060
device = torch.device('cpu')
print("Using CPU inference (two 8GB models, 8GB VRAM)")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def find_best_ckpt(glob_pattern):
    ckpts = glob.glob(glob_pattern)
    if not ckpts:
        sys.exit(f"ERROR: No checkpoint found at {glob_pattern}")
    return sorted(ckpts, key=lambda p: float(
        p.split('val_loss=')[1].split('.ckpt')[0]))[0]


def load_best_cpu(pth_path, ckpt_path):
    print(f"  Loading {os.path.basename(pth_path)}...")
    model = torch.load(pth_path, map_location='cpu', weights_only=False)
    print(f"  Loading {os.path.basename(ckpt_path)}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def reconstruct_cpu(model, test_ds, batch_size=128):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    all_orig, all_recon = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            dat, _, _ = batch
            x = dat['gex']
            xhat, _, _, _, _ = model.forward([x])
            all_orig.append(x.numpy())
            all_recon.append(xhat[0].numpy())
            if i % 20 == 0:
                print(f"    batch {i}/{len(loader)}", flush=True)
    return np.concatenate(all_orig), np.concatenate(all_recon)


def per_gene_rho(orig, recon):
    rhos = []
    n_genes = orig.shape[1]
    for j in range(n_genes):
        if np.std(orig[:, j]) > 0:
            rho, _ = stats.spearmanr(orig[:, j], recon[:, j])
            rhos.append(rho)
        else:
            rhos.append(np.nan)
        if j % 2000 == 0:
            print(f"    gene {j}/{n_genes}", flush=True)
    return np.array(rhos)


def per_sample_rho(orig, recon):
    return np.array([stats.spearmanr(orig[i], recon[i])[0]
                     for i in range(orig.shape[0])])


def impute(model, orig, fracs=(0.1, 0.2, 0.3), n_test=500, seed=42):
    # Matches evaluate_model() in train_denoising_vae.py exactly:
    # first n_test samples, per-sample random masking via torch.randperm, global rho.
    # torch.manual_seed makes the gene selection reproducible (locks the numbers).
    n_genes = orig.shape[1]
    n_test = min(n_test, orig.shape[0])
    gex_tensor = torch.tensor(orig, dtype=torch.float32)
    torch.manual_seed(seed)
    results = {}
    for frac in fracs:
        n_mask = int(n_genes * frac)
        masked_orig_all, masked_pred_all = [], []
        with torch.no_grad():
            for i in range(n_test):
                x = gex_tensor[i:i+1]
                idx = torch.randperm(n_genes)[:n_mask]
                x_masked = x.clone()
                x_masked[0, idx] = 0.0
                xhat, _, _, _, _ = model.forward([x_masked])
                masked_orig_all.append(x[0, idx].numpy())
                masked_pred_all.append(xhat[0][0, idx].numpy())
        orig_flat = np.concatenate(masked_orig_all)
        pred_flat = np.concatenate(masked_pred_all)
        rho, _ = stats.spearmanr(orig_flat, pred_flat)
        results[frac] = float(rho)
        print(f"    {int(frac*100)}% masking: rho={rho:.4f}")
    return results


# ----------------------------------------------------------------------
# Load data via HDF5
# ----------------------------------------------------------------------
print(f"\n[1] Loading test data from {DATA_PATH}...")
di = H5DataImporter(path=DATA_PATH, data_types=['gex'],
                    log_transform=False, top_percentile=100, min_features=100)
_, test_ds = di.import_data()
print(f"  Test samples: {len(test_ds)}")

# ----------------------------------------------------------------------
# Standard VAE
# ----------------------------------------------------------------------
print("\n[2] Standard VAE...")
std_ckpt = find_best_ckpt(STD_CKPT)
print(f"  Best Standard checkpoint: {os.path.basename(std_ckpt)}")
model_std = load_best_cpu(STD_PTH, std_ckpt)

print("  Reconstructing...")
orig_std, recon_std = reconstruct_cpu(model_std, test_ds)

print("  Per-gene rho...")
gene_rho_std = per_gene_rho(orig_std, recon_std)
print("  Per-sample rho...")
samp_rho_std = per_sample_rho(orig_std, recon_std)

print("  Imputation...")
imp_std = impute(model_std, orig_std)

del model_std; gc.collect()
print(f"  Standard done: gene_median={np.nanmedian(gene_rho_std):.4f}  "
      f"sample_median={np.nanmedian(samp_rho_std):.4f}")

# ----------------------------------------------------------------------
# Denoising VAE
# ----------------------------------------------------------------------
print("\n[3] Denoising VAE...")
dn_ckpt = find_best_ckpt(DN_CKPT)
print(f"  Best Denoising checkpoint: {os.path.basename(dn_ckpt)}")
model_dn = load_best_cpu(DN_PTH, dn_ckpt)

print("  Reconstructing...")
orig_dn, recon_dn = reconstruct_cpu(model_dn, test_ds)

print("  Per-gene rho...")
gene_rho_dn = per_gene_rho(orig_dn, recon_dn)
print("  Per-sample rho...")
samp_rho_dn = per_sample_rho(orig_dn, recon_dn)

print("  Imputation...")
imp_dn = impute(model_dn, orig_dn)

del model_dn; gc.collect()
print(f"  Denoising done: gene_median={np.nanmedian(gene_rho_dn):.4f}  "
      f"sample_median={np.nanmedian(samp_rho_dn):.4f}")

# ----------------------------------------------------------------------
# Save arrays for Fig S2 and downstream reuse
# ----------------------------------------------------------------------
np.savez_compressed(ARRAYS_NPZ,
    gene_rho_std=gene_rho_std, samp_rho_std=samp_rho_std,
    gene_rho_dn=gene_rho_dn,  samp_rho_dn=samp_rho_dn,
    orig_std=orig_std[:100],  recon_std=recon_std[:100],  # subset for FigS2
    orig_dn=orig_dn[:100],    recon_dn=recon_dn[:100],
    imp_std=np.array([imp_std[f] for f in [0.1, 0.2, 0.3]]),
    imp_dn=np.array([imp_dn[f]  for f in [0.1, 0.2, 0.3]]),
)
print(f"  Saved arrays: {ARRAYS_NPZ}")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"v3 RECONSTRUCTION SUMMARY (118k Scope B)")
print(f"{'='*55}")
print(f"  Standard  per-gene rho  : {np.nanmedian(gene_rho_std):.4f}  "
      f"(v2: 0.9138, +{(np.nanmedian(gene_rho_std)-0.9138)*100:.1f}%)")
print(f"  Denoising per-gene rho  : {np.nanmedian(gene_rho_dn):.4f}  "
      f"(v2: 0.9192, +{(np.nanmedian(gene_rho_dn)-0.9192)*100:.1f}%)")
print(f"  Standard  per-sample rho: {np.nanmedian(samp_rho_std):.4f}  "
      f"(v2: 0.7662, +{(np.nanmedian(samp_rho_std)-0.7662)*100:.1f}%)")
print(f"  Denoising per-sample rho: {np.nanmedian(samp_rho_dn):.4f}  "
      f"(v2: 0.7729, +{(np.nanmedian(samp_rho_dn)-0.7729)*100:.1f}%)")
for f in [0.1, 0.2, 0.3]:
    print(f"  Imputation {int(f*100)}%: std={imp_std[f]:.4f}  dn={imp_dn[f]:.4f}")
print(f"{'='*55}")

# ----------------------------------------------------------------------
# Plot — same style as v2 Fig 3
# ----------------------------------------------------------------------
print(f"\n[4] Plotting Fig 3...")
STD_COL = '#BBBBBB'
DEN_COL = '#9B59B6'
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel a: per-gene
ax = axes[0]
gene_std_clean = gene_rho_std[~np.isnan(gene_rho_std)]
gene_dn_clean  = gene_rho_dn[~np.isnan(gene_rho_dn)]
ax.hist(gene_std_clean, bins=60, color=STD_COL, alpha=0.85,
        label=f'Standard (med={np.nanmedian(gene_rho_std):.3f})')
ax.hist(gene_dn_clean,  bins=60, color=DEN_COL, alpha=0.75,
        label=f'Denoising (med={np.nanmedian(gene_rho_dn):.3f})')
ax.set_xlabel('Per-gene Spearman ρ', fontsize=11)
ax.set_ylabel('Number of genes', fontsize=11)
ax.legend(fontsize=9)
ax.text(-0.08, 1.04, 'a', transform=ax.transAxes,
        fontsize=16, fontweight='bold')

# Panel b: per-sample
ax = axes[1]
ax.hist(samp_rho_std, bins=60, color=STD_COL, alpha=0.85,
        label=f'Standard (med={np.nanmedian(samp_rho_std):.3f})')
ax.hist(samp_rho_dn,  bins=60, color=DEN_COL, alpha=0.75,
        label=f'Denoising (med={np.nanmedian(samp_rho_dn):.3f})')
ax.set_xlabel('Per-sample Spearman ρ', fontsize=11)
ax.set_ylabel('Number of samples', fontsize=11)
ax.legend(fontsize=9)
ax.text(-0.08, 1.04, 'b', transform=ax.transAxes,
        fontsize=16, fontweight='bold')

# Panel c: imputation bars
ax = axes[2]
fracs = [0.1, 0.2, 0.3]
x = np.arange(len(fracs))
w = 0.35
std_v = [imp_std[f] for f in fracs]
dn_v  = [imp_dn[f]  for f in fracs]
b1 = ax.bar(x - w/2, std_v, w, color=STD_COL, label='Standard')
b2 = ax.bar(x + w/2, dn_v,  w, color=DEN_COL, label='Denoising')
for bar, val in zip(list(b1)+list(b2), std_v+dn_v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(['10%', '20%', '30%'], fontsize=10)
ax.set_xlabel('Masking fraction', fontsize=11)
ax.set_ylabel('Imputation Spearman ρ', fontsize=11)
ax.set_ylim(min(std_v+dn_v)-0.02, max(std_v+dn_v)+0.025)
ax.legend(fontsize=9)
ax.text(-0.08, 1.04, 'c', transform=ax.transAxes,
        fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_SVG, dpi=300, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
print(f"  Saved: {OUT_SVG} / {OUT_PNG}")
