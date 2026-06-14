#!/usr/bin/env python3
"""
predict_tissue.py — Predict tissue types from a gene-expression CSV using the
compressed TorchScript model (no flexynesis dependency required).

Usage:
    python scripts/predict_tissue.py input.csv
    python scripts/predict_tissue.py input.csv --model-dir model_compressed --out predictions.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import torch

MODEL_DIR = Path("model_compressed")
ARTIFACTS  = Path("model_compressed/vae_tissue.artifacts.joblib")

def main():
    parser = argparse.ArgumentParser(description="Tissue-type prediction from gene expression")
    parser.add_argument("input", help="Gene-expression CSV (samples × genes or genes × samples)")
    parser.add_argument("--model-dir",  default=str(MODEL_DIR),
                        help="Directory with TorchScript model and label_mapping.json")
    parser.add_argument("--artifacts",  default=str(ARTIFACTS),
                        help="Path to vae_tissue.artifacts.joblib (for gene list and scaler)")
    parser.add_argument("--out",        default="predictions.csv",
                        help="Output CSV path (default: predictions.csv)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # ── Load model ────────────────────────────────────────────────────────────
    ts_path = model_dir / "vae_tissue_int8.torchscript.pt"
    model = torch.jit.load(str(ts_path))
    model.eval()

    # ── Load gene list, scaler, label mapping ─────────────────────────────────
    art          = joblib.load(args.artifacts)
    gene_list    = list(art["feature_lists"]["gex"])
    scaler       = art["transforms"]["gex"]
    label_mapping = {int(k): v for k, v in
                     json.loads((model_dir / "label_mapping.json").read_text()).items()}

    # ── Read and align input CSV ───────────────────────────────────────────────
    df = pd.read_csv(args.input, index_col=0)
    overlap_cols = len(set(df.columns) & set(gene_list))
    overlap_rows = len(set(df.index)   & set(gene_list))
    if overlap_rows > overlap_cols:
        df = df.T
    overlap = len(set(df.columns) & set(gene_list))
    print(f"Samples: {len(df)}   Gene overlap: {overlap}/{len(gene_list)} ({100*overlap/len(gene_list):.1f}%)")

    aligned = pd.DataFrame(0.0, index=df.index, columns=gene_list)
    common = [g for g in gene_list if g in df.columns]
    aligned[common] = df[common].values

    X = torch.tensor(scaler.transform(aligned.values), dtype=torch.float32)

    # ── Predict ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(X)

    nan_idx = {v: k for k, v in label_mapping.items()}.get("nan")
    if nan_idx is not None:
        logits[:, nan_idx] = -1e9

    probs       = torch.softmax(logits, dim=1)
    pred_idx    = logits.argmax(dim=1)
    pred_labels = [label_mapping[int(i)] for i in pred_idx]
    conf        = probs.max(dim=1).values.numpy()

    # ── Save results ──────────────────────────────────────────────────────────
    results = pd.DataFrame({
        "Sample":     df.index,
        "Tissue":     pred_labels,
        "Confidence": [f"{c:.1%}" for c in conf],
    })
    results.to_csv(args.out, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
