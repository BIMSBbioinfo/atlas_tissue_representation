# Flexynesis tissue-VAE manuscript — code and reproduction

Code, figures, and reproduction instructions for the manuscript:

**"Tissue-supervised latent representations from a curated 118K-sample multi-source bulk RNA-seq compendium"**
A. Pande, B. Uyar, A. Akalin (MDC Berlin / BIMSB)

This repository contains the analysis code and figure-generation scripts. Large
files (model weights, the HDF5 training compendium, pre-computed embeddings) are
hosted on Zenodo and linked below — they exceed GitHub's file-size limits.

---

## What is here

```
flexynesis_tissue_vae_manuscript/
├── README.md                  ← this file
├── TUTORIAL.md                ← how to use the pre-trained model (get embeddings + predictions)
├── LICENSE                    ← MIT (code)
├── scripts/                   ← all analysis + figure-generation code
│   ├── csv_to_h5.py                  build the HDF5 training compendium from CSVs
│   ├── h5_dataloader.py              memory-safe HDF5 dataloader (see also PR #146)
│   ├── train_denoising_vae_h5.py     train the supervised VAE (standard + denoising)
│   ├── train_denoising_vae.py        DenoisingVAE class + evaluation routine
│   ├── baseline_hvg_knn_v3.py        HVG + kNN baseline for comparison
│   ├── build_v3_webapp_artifacts.py  export embeddings/artifacts for the demo app
│   ├── get_embeddings.py             apply the pre-trained model to new data (inference)
│   ├── regen_fig1_v3.py              Figure 1  (t-SNE latent space)
│   ├── regen_fig2_final_v3.py        Figure 2  (per-class accuracy)
│   ├── regen_fig3_v3.py              Figure 3  (reconstruction + imputation)
│   ├── regen_fig4_v3.py              Figure 4  (TARGET developmental transfer)
│   ├── regen_fig5_v3.py              Figure 5  (single-cell foundation-model comparison)
│   ├── regen_figS1_v3.py             Figure S1 (confusion matrix)
│   ├── regen_figS2_v3.py             Figure S2 (per-gene reconstruction scatter)
│   └── regen_fig4_ablation.py        Figure S3 (cell-line ablation on TARGET)
├── figures/                   ← final figures (SVG + PNG, 300 DPI)
└── webapp/                    ← HuggingFace demo app (app.py, requirements, sample input)
```

> Note: the HDF5 dataloader is also contributed upstream to the Flexynesis
> package: https://github.com/BIMSBbioinfo/flexynesis/pull/146

Figure 6 (BulkFormer comparison). Figure 6 compares the supervised VAE against BulkFormer (Kang et al., bioRxiv 2025, doi:10.1101/2025.06.11.659222) in a zero-shot setting, using k-NN (k = 5, cosine) on BulkFormer embeddings of the TARGET cohort and the held-out reference set. The pre-computed BulkFormer embeddings are deposited on Zenodo.

---

## Data and model weights (Zenodo)

The following are deposited on Zenodo (DOI: **[to be added]**):

| Item | Size | Description |
|---|---|---|
| `processed_scaled_411k_tissue_B_h5/` | ~9 GB | HDF5 training compendium — 118,263 train / 28,274 test, 42 UBERON tissues, 16,115 genes |
| `results_denoising_vae_411k_B/` | ~18 GB | Standard + Denoising VAE weights, checkpoints, `results.json`, `target_v3_results.json` |
| `vae_tissue.final_model.pth` | ~8 GB | Trained model weights (used by the demo app) |
| pre-computed embeddings (`embeddings_{train,test}.csv`) | ~190 MB | Latent representations for downstream use |
| BulkFormer embeddings (`ref_emb_bf93M.npy`, `tgt_emb_bf93M.npy`) | ~60 MB | BulkFormer-93M embeddings of reference + TARGET (Figure 6) |

Download these from Zenodo and place them as indicated in the reproduction steps below.

---

## Environment

```bash
mamba create -n flexynesis python==3.11
mamba activate flexynesis
pip install flexynesis
```

Key dependencies (installed automatically via `pip install flexynesis`):
PyTorch, PyTorch Lightning, h5py (>=3.10), scikit-learn, pandas, numpy, matplotlib.

---

## Reproducing the analyses

> Paths below assume the Zenodo data has been downloaded into the repository
> root. Adjust paths in the scripts' configuration blocks if your layout differs.

**1. Build the HDF5 compendium** (only if rebuilding from source CSVs; otherwise
use the deposited HDF5 directly):
```bash
python scripts/csv_to_h5.py
```

**2. Train the VAE** (standard + denoising variants):
```bash
python scripts/train_denoising_vae_h5.py \
    --data_path processed_scaled_411k_tissue_B_h5 \
    --outdir results_denoising_vae_411k_B \
    --also_train_standard
```

**3. Regenerate figures** (each script reads the deposited model/results and
writes SVG + PNG into `figures/`):
```bash
python scripts/regen_fig1_v3.py        # Figure 1
python scripts/regen_fig2_final_v3.py  # Figure 2
python scripts/regen_fig3_v3.py        # Figure 3
python scripts/regen_fig4_v3.py        # Figure 4
python scripts/regen_fig5_v3.py        # Figure 5
python scripts/regen_figS1_v3.py       # Figure S1
python scripts/regen_figS2_v3.py       # Figure S2
python scripts/regen_fig4_ablation.py  # Figure S3 (cell-line ablation)
```

**4. Baseline comparison:**
```bash
python scripts/baseline_hvg_knn_v3.py
```

---

## Using the pre-trained model

To apply the trained model to your own bulk RNA-seq and get tissue embeddings + predictions, see [TUTORIAL.md](TUTORIAL.md) (or run `python get_embeddings.py`). Model weights and artifacts are on Zenodo (doi:10.5281/zenodo.20595537).

## Demo app

A live demo runs the trained model on user-supplied bulk RNA-seq:
https://huggingface.co/spaces/akalinLab/flexynesis-tissue-vae

The `webapp/` folder contains the app source and a small sample input
(`test_5samples.csv`).

---

## Licence

- Code (`scripts/`, `webapp/`) — MIT (see `LICENSE`)
- Data and model weights Zenodo (DOI: 10.5281/zenodo.20595537) — CC-BY-4.0

---

## Contact

Amit Pande — MDC Berlin / BIMSB — amit.pande@mdc-berlin.de
