#!/usr/bin/env python3
"""
compress_model.py — Strip embedded training data, quantize, and save a
deployable TorchScript bundle for the VAE tissue predictor.

Why the original is 7.5 GB:
  The model was saved with torch.save(model, ...) which pickled the entire
  Lightning object including model.dataset — a 118k×16k float32 tensor (~7.6 GB).
  Actual weights are only ~0.42 GB.

What this script produces in --out-dir:
  vae_tissue_int8.torchscript.pt   — quantized TorchScript (no flexynesis needed)
  label_mapping.json               — int→tissue-name lookup
  validation_predictions.csv       — predictions on --validate CSV (if given)

Usage:
  python scripts/compress_model.py --validate test_10samples.csv
  python scripts/compress_model.py --model model/vae_tissue.final_model.pth \\
                                   --artifacts model/vae_tissue.artifacts.joblib \\
                                   --out-dir model_compressed \\
                                   --validate test_10samples.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── Inference wrapper ────────────────────────────────────────────────────────
# Bundles only the three sub-modules needed for prediction; leaves the full
# Lightning model (with its embedded training dataset) out of the picture.

class TissuePredictor(nn.Module):
    def __init__(self, encoder: nn.Module, fc_mean: nn.Module, mlp: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.fc_mean  = fc_mean
        self.mlp      = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder returns (mean, log_var); we take the mean
        mean, _ = self.encoder(x)
        mu      = self.fc_mean(mean)
        return self.mlp(mu)


# ── Helpers ──────────────────────────────────────────────────────────────────

def align_input(csv_path: str, gene_list: list, scaler) -> tuple:
    """Read a gene-expression CSV and return (X_tensor, sample_names)."""
    df = pd.read_csv(csv_path, index_col=0)
    # auto-detect orientation: genes as columns vs rows
    overlap_cols = len(set(df.columns) & set(gene_list))
    overlap_rows = len(set(df.index)   & set(gene_list))
    if overlap_rows > overlap_cols:
        df = df.T
    overlap = len(set(df.columns) & set(gene_list))
    print(f"    Gene overlap: {overlap}/{len(gene_list)} ({100*overlap/len(gene_list):.1f}%)")
    aligned = pd.DataFrame(0.0, index=df.index, columns=gene_list)
    common = [g for g in gene_list if g in df.columns]
    aligned[common] = df[common].values
    X = torch.tensor(scaler.transform(aligned.values), dtype=torch.float32)
    return X, list(df.index)


def predict(model: nn.Module, X: torch.Tensor, label_mapping: dict) -> pd.DataFrame:
    """Run forward pass and return a DataFrame with Tissue + confidence."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
    # mask the 'nan' class if present
    nan_idx = {v: k for k, v in label_mapping.items()}.get("nan")
    if nan_idx is not None:
        logits[:, int(nan_idx)] = -1e9
    probs       = torch.softmax(logits, dim=1)
    pred_idx    = logits.argmax(dim=1)
    pred_labels = [label_mapping[int(i)] for i in pred_idx]
    conf        = probs.max(dim=1).values.numpy()
    return pred_labels, conf


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compress VAE tissue predictor for deployment"
    )
    parser.add_argument("--model",     default="model/vae_tissue.final_model.pth",
                        help="Path to the original .pth model file")
    parser.add_argument("--artifacts", default="model_compressed/vae_tissue.artifacts.joblib",
                        help="Path to the artifacts .joblib file")
    parser.add_argument("--out-dir",   default="model_compressed",
                        help="Directory to write compressed model files")
    parser.add_argument("--validate",  default=None,
                        help="Gene-expression CSV to validate predictions on")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load original model ──────────────────────────────────────────────────
    print(f"\n[1] Loading model from {args.model}  (may take ~30 s for 7.5 GB file…)")
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model.eval()

    art           = joblib.load(args.artifacts)
    gene_list     = list(art["feature_lists"]["gex"])
    scaler        = art["transforms"]["gex"]
    label_mapping = model.dataset.label_mappings["uberon_tissue"]
    print(f"    Input genes: {len(gene_list)} | Output classes: {len(label_mapping)}")

    original_size_gb = Path(args.model).stat().st_size / 1e9
    print(f"    Original file size: {original_size_gb:.2f} GB")

    # Count actual weight bytes
    sd = model.state_dict()
    weight_gb = sum(v.numel() * v.element_size() for v in sd.values()) / 1e9
    print(f"    Weight-only footprint: {weight_gb:.3f} GB  "
          f"({weight_gb/original_size_gb:.1%} of file — rest is embedded training data)")

    # 2. Build the inference wrapper ──────────────────────────────────────────
    print("\n[2] Extracting inference sub-modules…")
    predictor = TissuePredictor(
        encoder=model.encoders[0],
        fc_mean=model.FC_mean,
        mlp=model.MLPs["uberon_tissue"],
    ).eval()

    # free the 7.5 GB model from memory now that we've extracted what we need
    del model
    import gc; gc.collect()

    # 3. int8 dynamic quantization ────────────────────────────────────────────
    print("[3] Applying int8 dynamic quantization to all Linear layers…")
    quantized = torch.ao.quantization.quantize_dynamic(
        predictor, {nn.Linear}, dtype=torch.qint8
    ).eval()

    # 4. Save as TorchScript bundle ───────────────────────────────────────────
    print("[4] Compiling to TorchScript…")
    ts_path = out_dir / "vae_tissue_int8.torchscript.pt"
    try:
        # Use trace (more permissive than script for third-party module classes)
        dummy = torch.zeros(1, len(gene_list))
        traced = torch.jit.trace(quantized, dummy)
        torch.jit.save(traced, str(ts_path))
        ts_size_gb = ts_path.stat().st_size / 1e9
        print(f"    Saved: {ts_path}  ({ts_size_gb:.3f} GB)")
        print(f"    Compression vs original:     {original_size_gb/ts_size_gb:.0f}x")
        print(f"    Compression vs weights-only: {weight_gb/ts_size_gb:.1f}x")
        deployed_model = torch.jit.load(str(ts_path))
        deployed_model.eval()
        print("    Load round-trip: OK")
    except Exception as e:
        print(f"    TorchScript failed: {e}")
        print("    Falling back to saving fp32 wrapper state dict…")
        fb_path = out_dir / "vae_tissue_fp32_inference.pt"
        torch.save(predictor.state_dict(), fb_path)
        print(f"    Saved state dict: {fb_path}")
        deployed_model = quantized
        ts_path = None

    # 5. Save label mapping ───────────────────────────────────────────────────
    meta_path = out_dir / "label_mapping.json"
    meta_path.write_text(
        json.dumps({str(k): v for k, v in label_mapping.items()}, indent=2)
    )
    print(f"\n[5] Saved label mapping ({len(label_mapping)} classes) → {meta_path}")

    # 6. Validate ─────────────────────────────────────────────────────────────
    if args.validate:
        print(f"\n[6] Validating on {args.validate}")
        X, samples = align_input(args.validate, gene_list, scaler)

        # fp32 predictor vs quantized
        with torch.no_grad():
            preds_fp32 = predictor(X).argmax(1)
            preds_q    = deployed_model(X).argmax(1)

        agreement = (preds_fp32 == preds_q).float().mean().item()
        status = "OK" if agreement >= 0.98 else "WARNING — below 0.98"
        print(f"    fp32 vs int8 argmax agreement: {agreement:.4f}  [{status}]")

        pred_labels, conf = predict(deployed_model, X, label_mapping)
        results = pd.DataFrame({
            "Sample": samples,
            "Tissue": pred_labels,
            "Confidence": [f"{c:.1%}" for c in conf],
        })
        val_path = out_dir / "validation_predictions.csv"
        results.to_csv(val_path, index=False)
        print(f"    Predictions saved → {val_path}")
        print()
        print(results.to_string(index=False))

    # 7. Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Output files in {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:<45}  {f.stat().st_size/1e6:>8.1f} MB")

    if ts_path and ts_path.exists():
        print(f"\nTo run inference with the compressed model (no flexynesis needed):")
        print(f"  loaded = torch.jit.load('{ts_path}')")
        print( "  loaded.eval()")
        print( "  logits = loaded(X_tensor)   # X: [n_samples, n_genes], float32")


if __name__ == "__main__":
    main()
