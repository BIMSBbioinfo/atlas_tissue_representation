#!/usr/bin/env python3
"""FigS2 — v3 per-gene scatter (Standard vs Denoising) — uses cached arrays."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARRAYS = 'results_denoising_vae_411k_B/fig3_arrays_v3.npz'

print(f"[1] Loading cached arrays from {ARRAYS}...")
data = np.load(ARRAYS)
std_rho = data['gene_rho_std']
dn_rho  = data['gene_rho_dn']

# Mask NaN (genes with zero variance)
mask = ~(np.isnan(std_rho) | np.isnan(dn_rho))
std_rho = std_rho[mask]
dn_rho  = dn_rho[mask]
n_genes = len(std_rho)

n_improved = int(np.sum(dn_rho > std_rho))
med_std    = float(np.median(std_rho))
med_dn     = float(np.median(dn_rho))

print(f"    Genes:           {n_genes}")
print(f"    Median Standard: {med_std:.4f}")
print(f"    Median Denoising:{med_dn:.4f}")
print(f"    Improved:        {n_improved}/{n_genes} ({100*n_improved/n_genes:.1f}%)")

print(f"\n[2] Plotting...")
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(std_rho, dn_rho, alpha=0.15, s=2, color='#6A0DAD', rasterized=True)
lims = [min(std_rho.min(), dn_rho.min()) - 0.02,
        max(std_rho.max(), dn_rho.max()) + 0.02]
ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5, label='y = x')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Standard VAE (Spearman ρ)', fontsize=12)
ax.set_ylabel('Denoising VAE (Spearman ρ)', fontsize=12)
ax.set_title('Per-gene reconstruction: Standard vs Denoising VAE (v3, 118K)',
             fontsize=11, fontweight='bold')
ax.text(0.05, 0.95,
        f'Median ρ: Standard = {med_std:.3f}, Denoising = {med_dn:.3f}\n'
        f'{n_improved:,}/{n_genes:,} genes improved ({100*n_improved/n_genes:.1f}%)',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('FigS2_pergene_scatter_v3_118k.svg', dpi=300, bbox_inches='tight')
plt.savefig('FigS2_pergene_scatter_v3_118k.png', dpi=300, bbox_inches='tight')
print(f"    Saved: FigS2_pergene_scatter_v3_118k.{{svg,png}}")
print(f"\nDone.")
