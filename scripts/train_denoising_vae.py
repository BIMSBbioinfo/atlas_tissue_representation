#!/usr/bin/env python3
"""
Denoising VAE Training for Flexynesis
======================================
Subclasses supervised_vae to add random gene masking during training.
Uses Flexynesis DataImporter for proper data loading and normalization.

Usage:
    python train_denoising_vae.py
    python train_denoising_vae.py --mask_fraction 0.3 --also_train_standard

Author: Amit Pande, MDC Berlin/BIMSB
"""

import argparse
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import DataLoader
from scipy import stats
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from flexynesis.data import DataImporter
from flexynesis.models.supervised_vae import supervised_vae


class DenoisingVAE(supervised_vae):
    """
    Supervised VAE with random gene masking during training.
    Forward pass receives masked input; reconstruction loss against originals.
    Validation/inference: no masking.
    """

    def __init__(self, config, dataset, target_variables, mask_fraction=0.2,
                 batch_variables=None, surv_event_var=None, surv_time_var=None,
                 use_loss_weighting=False, device_type=None):
        super().__init__(config, dataset, target_variables,
                         batch_variables=batch_variables,
                         surv_event_var=surv_event_var,
                         surv_time_var=surv_time_var,
                         use_loss_weighting=use_loss_weighting,
                         device_type=device_type)
        self.mask_fraction = mask_fraction
        print(f"[DenoisingVAE] Mask fraction: {mask_fraction}")

    def _apply_mask(self, x):
        """Random binary mask: zeros out mask_fraction of genes per sample."""
        mask = torch.ones_like(x)
        n_genes = x.shape[1]
        n_mask = int(n_genes * self.mask_fraction)
        for i in range(x.shape[0]):
            idx = torch.randperm(n_genes, device=x.device)[:n_mask]
            mask[i, idx] = 0.0
        return x * mask

    def MMD_loss_denoising(self, latent_dim, z, xhat, x_original):
        """MMD + reconstruction loss against ORIGINAL (not masked) input."""
        true_samples = torch.randn(200, latent_dim, device=self.device)
        mmd = self.compute_mmd(true_samples, z)
        nll = (xhat - x_original).pow(2).mean()
        return mmd + nll

    def training_step(self, train_batch, batch_idx, log=True):
        dat, y_dict, samples = train_batch
        layers = list(dat.keys())

        # Save originals, mask inputs
        x_list_original = [dat[x] for x in layers]
        x_list_masked = [self._apply_mask(x) for x in x_list_original]

        # Forward with masked input
        x_hat_list, z, mean, log_var, outputs = self.forward(x_list_masked)

        # Reconstruction loss against originals
        mmd_loss_list = [
            self.MMD_loss_denoising(z.shape[1], z, x_hat_list[i], x_list_original[i])
            for i in range(len(layers))
        ]
        mmd_loss = torch.mean(torch.stack(mmd_loss_list))

        # Supervisor losses (unchanged)
        losses = {'mmd_loss': mmd_loss}
        for var in self.variables:
            if var == self.surv_event_var:
                durations = y_dict[self.surv_time_var]
                events = y_dict[self.surv_event_var]
                risk_scores = outputs[var]
                from flexynesis.modules import cox_ph_loss
                loss = cox_ph_loss(risk_scores, durations, events)
            else:
                y_hat = outputs[var]
                y = y_dict[var]
                loss = self.compute_loss(var, y, y_hat)
            losses[var] = loss

        total_loss = self.compute_total_loss(losses)
        losses['train_loss'] = total_loss
        if log:
            self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    # validation_step inherited from supervised_vae (no masking)


def evaluate_model(model, test_ds, device, label=""):
    """Full evaluation: reconstruction + imputation."""
    model.eval()
    model.to(device)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_orig, all_recon = [], []
    with torch.no_grad():
        for batch in loader:
            dat, _, _ = batch
            x = dat['gex'].to(device)
            x_hat_list, _, _, _, _ = model.forward([x])
            all_orig.append(x.cpu().numpy())
            all_recon.append(x_hat_list[0].cpu().numpy())

    orig = np.concatenate(all_orig)
    recon = np.concatenate(all_recon)

    # Per-gene Spearman
    gene_rhos = []
    for j in range(orig.shape[1]):
        if np.std(orig[:, j]) > 0:
            rho, _ = stats.spearmanr(orig[:, j], recon[:, j])
            gene_rhos.append(rho)

    # Per-sample Spearman
    sample_rhos = []
    for i in range(orig.shape[0]):
        rho, _ = stats.spearmanr(orig[i], recon[i])
        sample_rhos.append(rho)

    print(f"\n  {label} Reconstruction:")
    print(f"    Per-gene rho:   {np.nanmedian(gene_rhos):.4f} (median)")
    print(f"    Per-sample rho: {np.nanmedian(sample_rhos):.4f} (median)")

    # Gene imputation at different masking levels
    print(f"  {label} Gene Imputation:")
    n_genes = orig.shape[1]
    n_test = min(500, orig.shape[0])
    imputation = {}

    # Get the tensor for imputation
    gex_tensor = torch.tensor(orig, dtype=torch.float32)

    for frac in [0.1, 0.2, 0.3]:
        n_mask = int(n_genes * frac)
        masked_orig_all, masked_pred_all = [], []
        with torch.no_grad():
            for i in range(n_test):
                x = gex_tensor[i:i+1].to(device)
                idx = torch.randperm(n_genes)[:n_mask]
                x_masked = x.clone()
                x_masked[0, idx] = 0.0
                x_hat_list, _, _, _, _ = model.forward([x_masked])
                masked_orig_all.append(x[0, idx].cpu().numpy())
                masked_pred_all.append(x_hat_list[0][0, idx].cpu().numpy())
        orig_flat = np.concatenate(masked_orig_all)
        pred_flat = np.concatenate(masked_pred_all)
        rho, _ = stats.spearmanr(orig_flat, pred_flat)
        imputation[frac] = rho
        print(f"    {int(frac*100)}% masking: rho = {rho:.4f}")

    return {
        'per_gene_rho': np.nanmedian(gene_rhos),
        'per_sample_rho': np.nanmedian(sample_rhos),
        'imputation': imputation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="processed_scaled_50k_tissue")
    parser.add_argument("--mask_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--outdir", default="results_denoising_vae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--also_train_standard", action="store_true",
                        help="Also train standard VAE for direct comparison")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # Best config from tissue VAE HPO
    config = {
        'latent_dim': 121,
        'hidden_dim_factor': 0.2002336297523043,
        'supervisor_hidden_dim': 32,
        'lr': 0.0017177621112338384,
        'batch_size': 32,
    }

    print("=" * 70)
    print("DENOISING VAE EXPERIMENT")
    print("=" * 70)
    print(f"Config: latent_dim={config['latent_dim']}, lr={config['lr']:.6f}")
    print(f"Mask fraction: {args.mask_fraction}")

    # Load data through Flexynesis pipeline (handles scaling, NA imputation)
    print("\n1. Loading data via Flexynesis DataImporter...")
    di = DataImporter(
        path=args.data_path,
        data_types=['gex'],
        log_transform=False,
        top_percentile=100,
        min_features=100,
    )
    train_ds, test_ds = di.import_data()
    print(f"   Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"   Genes: {len(train_ds.features['gex'])}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- Train Denoising VAE ---
    print(f"\n2. Training Denoising VAE (mask={args.mask_fraction})...")
    model_dn = DenoisingVAE(
        config=config,
        dataset=train_ds,
        target_variables=['uberon_tissue'],
        mask_fraction=args.mask_fraction,
        use_loss_weighting=False,
        device_type=args.device,
    )
    print(f"   Parameters: {sum(p.numel() for p in model_dn.parameters()):,}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=3)
    val_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=3)

    callbacks_dn = [
        EarlyStopping(monitor="val_loss", patience=args.early_stop_patience, mode="min"),
        ModelCheckpoint(
            dirpath=str(outdir / "ckpt_denoising"),
            filename="best_{epoch}_{val_loss:.4f}",
            monitor="val_loss", save_top_k=1, mode="min"
        ),
    ]

    trainer_dn = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks_dn,
        accelerator="gpu" if "cuda" in args.device else "cpu",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )
    trainer_dn.fit(model_dn, train_loader, val_loader)

    # Evaluate
    print(f"\n3. Evaluating Denoising VAE...")
    results_dn = evaluate_model(model_dn, test_ds, device, label="[Denoising]")

    # Save model
    torch.save(model_dn, outdir / "denoising_vae.pth")

    # --- Optionally train standard VAE ---
    results_std = None
    if args.also_train_standard:
        print(f"\n4. Training Standard VAE (no masking)...")
        model_std = supervised_vae(
            config=config,
            dataset=train_ds,
            target_variables=['uberon_tissue'],
            use_loss_weighting=False,
            device_type=args.device,
        )

        callbacks_std = [
            EarlyStopping(monitor="val_loss", patience=args.early_stop_patience, mode="min"),
            ModelCheckpoint(
                dirpath=str(outdir / "ckpt_standard"),
                filename="best_{epoch}_{val_loss:.4f}",
                monitor="val_loss", save_top_k=1, mode="min"
            ),
        ]

        trainer_std = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=callbacks_std,
            accelerator="gpu" if "cuda" in args.device else "cpu",
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=50,
        )
        trainer_std.fit(model_std, train_loader, val_loader)

        print(f"\n5. Evaluating Standard VAE...")
        results_std = evaluate_model(model_std, test_ds, device, label="[Standard]")
        torch.save(model_std, outdir / "standard_vae.pth")

    # --- Print comparison ---
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<35} ", end="")
    if results_std:
        print(f"{'Standard':>12} {'Denoising':>12}")
    else:
        print(f"{'Denoising':>12}")
    print("-" * (50 if results_std else 48))
    
    if results_std:
        print(f"{'Per-gene recon rho':<35} {results_std['per_gene_rho']:>12.4f} {results_dn['per_gene_rho']:>12.4f}")
        print(f"{'Per-sample recon rho':<35} {results_std['per_sample_rho']:>12.4f} {results_dn['per_sample_rho']:>12.4f}")
        for frac in [0.1, 0.2, 0.3]:
            print(f"{'Imputation @ ' + str(int(frac*100)) + '%':<35} {results_std['imputation'][frac]:>12.4f} {results_dn['imputation'][frac]:>12.4f}")
    else:
        print(f"{'Per-gene recon rho':<35} {results_dn['per_gene_rho']:>12.4f}")
        print(f"{'Per-sample recon rho':<35} {results_dn['per_sample_rho']:>12.4f}")
        for frac in [0.1, 0.2, 0.3]:
            print(f"{'Imputation @ ' + str(int(frac*100)) + '%':<35} {results_dn['imputation'][frac]:>12.4f}")

    # Save JSON
    all_results = {'denoising': results_dn, 'config': config, 'mask_fraction': args.mask_fraction}
    if results_std:
        all_results['standard'] = results_std
    # Convert numpy to float for JSON
    def to_json(obj):
        if isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    with open(outdir / "results.json", 'w') as f:
        json.dump(to_json(all_results), f, indent=2)
    print(f"\nResults saved to {outdir}/results.json")


if __name__ == "__main__":
    main()
