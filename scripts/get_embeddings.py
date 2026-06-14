#!/usr/bin/env python3
"""
get_embeddings.py — Get 121-dim latent embeddings and tissue predictions from the
FULL pre-trained VAE (vae_tissue.final_model.pth, deposited on Zenodo).

Requires the flexynesis environment (the model is a supervised_vae subclass).
For the lightweight, flexynesis-free version, see scripts/predict_tissue.py.

Usage:
    python scripts/get_embeddings.py input.csv
    python scripts/get_embeddings.py input.csv --model-dir model --out-prefix myrun
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib


def main():
    parser = argparse.ArgumentParser(
        description="Latent embeddings + tissue predictions from the full VAE")
    parser.add_argument("input", help="Gene-expression CSV (genes×samples or samples×genes, HGNC symbols, log2)")
    parser.add_argument("--model-dir", default="model",
                        help="Directory with vae_tissue.final_model.pth and vae_tissue.artifacts.joblib (default: model)")
    parser.add_argument("--out-prefix", default="output",
                        help="Prefix for output files: <prefix>_predictions.csv and <prefix>_embeddings.csv (default: output)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # ── Load model and artifacts ──────────────────────────────────────────────
    model = torch.load(model_dir / "vae_tissue.final_model.pth",
                       map_location="cpu", weights_only=False)
    model.eval()
    art = joblib.load(model_dir / "vae_tissue.artifacts.joblib")
    gene_list = list(art["feature_lists"]["gex"])
    scaler = art["transforms"]["gex"]

    # ── Read and auto-orient the input ────────────────────────────────────────
    df = pd.read_csv(args.input, index_col=0)
    gic = len(set(df.columns) & set(gene_list))
    gir = len(set(df.index) & set(gene_list))
    if gir > gic:
        df = df.T
    overlap = len(set(df.columns) & set(gene_list))
    print(f"Samples: {df.shape[0]}   Gene overlap: {overlap}/{len(gene_list)} "
          f"({100*overlap/len(gene_list):.1f}%)")

    # ── Align to model gene space (zero-fill missing) ─────────────────────────
    aligned = pd.DataFrame(0.0, index=df.index, columns=gene_list)
    common = [g for g in gene_list if g in df.columns]
    aligned[common] = df[common].values
    aligned = aligned.fillna(0)

    # ── Encode ────────────────────────────────────────────────────────────────
    X = torch.tensor(scaler.transform(aligned.values), dtype=torch.float32)
    with torch.no_grad():
        h = model.encoders[0](X)
        mu = model.FC_mean(h[0])
    emb = mu.numpy()
    print("Embeddings shape:", emb.shape)

    # ── Predict tissue ────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model.MLPs["uberon_tissue"](mu)
        lm = model.dataset.label_mappings["uberon_tissue"]
        n2i = {v: k for k, v in lm.items()}
        if "nan" in n2i:
            logits[:, n2i["nan"]] = -1e9
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
    labels = [lm[int(i)] for i in pred]
    conf = probs.max(dim=1).values.numpy()

    # ── Save outputs ──────────────────────────────────────────────────────────
    pred_path = f"{args.out_prefix}_predictions.csv"
    emb_path = f"{args.out_prefix}_embeddings.csv"

    results = pd.DataFrame({
        "Sample": df.index,
        "Tissue": labels,
        "Confidence": [f"{c:.1%}" for c in conf],
    })
    results.to_csv(pred_path, index=False)

    emb_df = pd.DataFrame(emb, index=df.index,
                          columns=[f"z{i}" for i in range(emb.shape[1])])
    emb_df.to_csv(emb_path)

    print(results.to_string(index=False))
    print(f"\nSaved → {pred_path}")
    print(f"Saved → {emb_path}")


if __name__ == "__main__":
    main()
