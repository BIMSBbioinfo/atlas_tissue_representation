#!/usr/bin/env python3
"""
Denoising VAE Training — HDF5 variant.

Identical to train_denoising_vae.py EXCEPT one line: uses H5DataImporter
(reads gex from HDF5 float32) instead of stock DataImporter (CSV float64).

This is the memory-safe path for 100k+ sample training compendia.

Usage (mirrors original script):
    python train_denoising_vae_h5.py \\
        --data_path processed_scaled_411k_tissue_B_h5 \\
        --outdir results_denoising_vae_411k_B \\
        --mask_fraction 0.2 \\
        --epochs 500 \\
        --early_stop_patience 10 \\
        --also_train_standard

Hyperparameters preserved EXACTLY from 75k_v2 HPO-optimized config
(latent_dim=121, lr=0.00172, batch_size=32) for direct ablation
comparability. Only data scale and modality differ.

Author: Amit Pande, MDC Berlin/BIMSB
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from flexynesis.models.supervised_vae import supervised_vae

# Re-use existing DenoisingVAE and evaluate_model from the original script
sys.path.insert(0, str(Path(__file__).parent))
from train_denoising_vae import DenoisingVAE, evaluate_model

# HDF5-backed importer
from h5_dataloader import H5DataImporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        default="processed_scaled_411k_tissue_B_h5")
    parser.add_argument("--mask_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--outdir", default="results_denoising_vae_411k_B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--also_train_standard", action="store_true",
                        help="Also train standard VAE for direct comparison")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 safest for HDF5 in-memory)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # ==================================================================
    # CONFIG: locked to 75k_v2 HPO-optimized values for ablation parity
    # ==================================================================
    config = {
        'latent_dim': 121,
        'hidden_dim_factor': 0.2002336297523043,
        'supervisor_hidden_dim': 32,
        'lr': 0.0017177621112338384,
        'batch_size': 32,
    }

    print("=" * 70)
    print("DENOISING VAE EXPERIMENT — HDF5 variant (v3 Scope B)")
    print("=" * 70)
    print(f"Config: latent_dim={config['latent_dim']}, lr={config['lr']:.6f}")
    print(f"Mask fraction: {args.mask_fraction}")
    print(f"Data path: {args.data_path}")
    print(f"Outdir:    {args.outdir}")

    # ------------------------------------------------------------------
    # 1. Load data via H5DataImporter (HDF5-backed)
    # ------------------------------------------------------------------
    print("\n1. Loading data via H5DataImporter...")
    di = H5DataImporter(
        path=args.data_path,
        data_types=['gex'],
        log_transform=False,
        top_percentile=100,
        min_features=100,
    )
    train_ds, test_ds = di.import_data()
    print(f"   Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"   Genes: {len(train_ds.features['gex'])}")
    print(f"   Tissues: {len(train_ds.label_mappings.get('uberon_tissue', {}))}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # 2. Train Denoising VAE
    # ------------------------------------------------------------------
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

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'],
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        test_ds, batch_size=config['batch_size'],
        shuffle=False, num_workers=args.num_workers,
    )

    callbacks_dn = [
        EarlyStopping(monitor="val_loss",
                      patience=args.early_stop_patience, mode="min"),
        ModelCheckpoint(
            dirpath=str(outdir / "ckpt_denoising"),
            filename="best_{epoch}_{val_loss:.4f}",
            monitor="val_loss", save_top_k=1, mode="min",
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

    print("\n3. Evaluating Denoising VAE...")
    results_dn = evaluate_model(model_dn, test_ds, device, label="[Denoising]")
    torch.save(model_dn, outdir / "denoising_vae.pth")

    # ------------------------------------------------------------------
    # 4. Optionally train Standard VAE for ablation comparison
    # ------------------------------------------------------------------
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
            EarlyStopping(monitor="val_loss",
                          patience=args.early_stop_patience, mode="min"),
            ModelCheckpoint(
                dirpath=str(outdir / "ckpt_standard"),
                filename="best_{epoch}_{val_loss:.4f}",
                monitor="val_loss", save_top_k=1, mode="min",
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
        results_std = evaluate_model(model_std, test_ds, device,
                                     label="[Standard]")
        torch.save(model_std, outdir / "standard_vae.pth")

    # ------------------------------------------------------------------
    # 6. Print + save comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY (v3 Scope B HDF5)")
    print(f"{'='*70}")
    print(f"{'Metric':<35} ", end="")
    if results_std:
        print(f"{'Standard':>12} {'Denoising':>12}")
    else:
        print(f"{'Denoising':>12}")
    print("-" * (50 if results_std else 48))

    if results_std:
        print(f"{'Per-gene recon rho':<35} "
              f"{results_std['per_gene_rho']:>12.4f} "
              f"{results_dn['per_gene_rho']:>12.4f}")
        print(f"{'Per-sample recon rho':<35} "
              f"{results_std['per_sample_rho']:>12.4f} "
              f"{results_dn['per_sample_rho']:>12.4f}")
        for frac in [0.1, 0.2, 0.3]:
            print(f"{'Imputation @ ' + str(int(frac*100)) + '%':<35} "
                  f"{results_std['imputation'][frac]:>12.4f} "
                  f"{results_dn['imputation'][frac]:>12.4f}")
    else:
        print(f"{'Per-gene recon rho':<35} {results_dn['per_gene_rho']:>12.4f}")
        print(f"{'Per-sample recon rho':<35} {results_dn['per_sample_rho']:>12.4f}")
        for frac in [0.1, 0.2, 0.3]:
            print(f"{'Imputation @ ' + str(int(frac*100)) + '%':<35} "
                  f"{results_dn['imputation'][frac]:>12.4f}")

    # JSON dump
    all_results = {
        'denoising': results_dn,
        'config': config,
        'mask_fraction': args.mask_fraction,
        'data_path': args.data_path,
        'n_train': len(train_ds),
        'n_test': len(test_ds),
        'n_genes': len(train_ds.features['gex']),
        'n_tissues': len(train_ds.label_mappings.get('uberon_tissue', {})),
    }
    if results_std:
        all_results['standard'] = results_std

    def to_json(obj):
        if isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(outdir / "results.json", 'w') as f:
        json.dump(to_json(all_results), f, indent=2)
    print(f"\nResults saved to {outdir}/results.json")


if __name__ == "__main__":
    main()
