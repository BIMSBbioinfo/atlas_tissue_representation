[TUTORIAL.md](https://github.com/user-attachments/files/28746445/TUTORIAL.md)
# Tutorial: Getting tissue embeddings and predictions from the Flexynesis tissue-VAE

This guide shows how to take a bulk RNA-seq expression matrix and obtain (1) the
121-dimensional latent embedding and (2) the predicted UBERON tissue of origin,
using the pre-trained model — **no retraining required**.

The trained model and its artifacts are deposited on Zenodo
(doi:10.5281/zenodo.20595537). Download these into a `model/` directory:

| File | Purpose |
|---|---|
| `vae_tissue.final_model.pth` | Trained model weights |
| `vae_tissue.artifacts.joblib` | Gene list, scaler, and label encoder |
| `embeddings_train.csv`, `embeddings_test.csv` | Reference embeddings (for kNN, optional) |
| `train_clin.csv`, `test_clin.csv` | Reference tissue labels (for kNN, optional) |

A small example input, `test_5samples.csv`, is included in the `webapp/` folder of
the repository.

---

## 1. Environment

```bash
conda env create -f environment.yml
conda activate flexynesis
```

Core dependencies: PyTorch, pandas, numpy, scikit-learn (1.7.x), joblib.

> Note: the model artifacts were saved with scikit-learn 1.7.2. Use a matching
> version to avoid an unpickling version warning.

---

## 2. Input format

- **Rows or columns** may be genes; the code auto-orients the matrix.
- **Gene identifiers:** HGNC gene symbols.
- **Values:** log2-transformed expression (TPM, RPKM, or counts).
- Genes not seen by the model are ignored; missing model genes are zero-filled.
  A gene overlap below ~1,000 makes results unreliable.

---

## 3. Minimal script: expression matrix → embeddings + tissue prediction

```python
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path

MODEL_DIR = Path("model")

# ---- Load model and artifacts ----
model = torch.load(MODEL_DIR / "vae_tissue.final_model.pth",
                   map_location="cpu", weights_only=False)
model.eval()

art       = joblib.load(MODEL_DIR / "vae_tissue.artifacts.joblib")
gene_list = list(art["feature_lists"]["gex"])      # genes the model expects
scaler    = art["transforms"]["gex"]               # fitted StandardScaler

# ---- Load your expression matrix (HGNC symbols, log2 expression) ----
df = pd.read_csv("test_5samples.csv", index_col=0)

# Auto-orient so that samples are rows, genes are columns
genes_in_cols = len(set(df.columns) & set(gene_list))
genes_in_rows = len(set(df.index)   & set(gene_list))
if genes_in_rows > genes_in_cols:
    df = df.T

overlap = len(set(df.columns) & set(gene_list))
print(f"Gene overlap: {overlap}/{len(gene_list)} "
      f"({100*overlap/len(gene_list):.1f}%)")

# ---- Align to the model's gene space (zero-fill missing genes) ----
aligned          = pd.DataFrame(0.0, index=df.index, columns=gene_list)
common           = [g for g in gene_list if g in df.columns]
aligned[common]  = df[common].values
aligned          = aligned.fillna(0)

# ---- Scale and run the encoder ----
X_scaled = scaler.transform(aligned)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

with torch.no_grad():
    h  = model.encoders[0](X_tensor)
    mu = model.FC_mean(h[0])          # 121-dimensional latent embedding

embeddings = mu.numpy()
print("Embeddings shape:", embeddings.shape)   # (n_samples, 121)

# ---- Tissue classification from the supervised head ----
with torch.no_grad():
    logits = model.MLPs["uberon_tissue"](mu)
    label_mapping = model.dataset.label_mappings["uberon_tissue"]
    name_to_idx   = {v: k for k, v in label_mapping.items()}
    nan_idx       = name_to_idx.get("nan", None)
    if nan_idx is not None:
        logits[:, nan_idx] = -1e9     # mask the 'nan' class if present
    probs    = torch.softmax(logits, dim=1)
    pred_idx = logits.argmax(dim=1)

pred_labels = [label_mapping[int(i)] for i in pred_idx]
confidence  = probs.max(dim=1).values.numpy()

# ---- Save ----
emb_df = pd.DataFrame(embeddings, index=df.index,
                      columns=[f"z{i}" for i in range(embeddings.shape[1])])
emb_df.to_csv("my_embeddings.csv")

results = pd.DataFrame({
    "Sample":     df.index,
    "Tissue":     pred_labels,
    "Confidence": [f"{c:.1%}" for c in confidence],
})
results.to_csv("my_predictions.csv", index=False)
print(results)
```

A ready-to-run version of this script is provided as `get_embeddings.py`.

---

## 4. Optional: kNN tissue label from reference embeddings

If you prefer a nearest-neighbour label over the supervised head (e.g. for
out-of-distribution samples), use the deposited reference embeddings:

```python
from sklearn.neighbors import KNeighborsClassifier

train_emb = pd.read_csv(MODEL_DIR / "embeddings_train.csv", index_col=0)
test_emb  = pd.read_csv(MODEL_DIR / "embeddings_test.csv",  index_col=0)
train_clin = pd.read_csv(MODEL_DIR / "train_clin.csv", index_col=0)
test_clin  = pd.read_csv(MODEL_DIR / "test_clin.csv",  index_col=0)

ref_emb  = pd.concat([train_emb, test_emb])
ref_clin = pd.concat([train_clin, test_clin])
idx      = ref_emb.index.intersection(ref_clin.index)
ref_emb, ref_clin = ref_emb.loc[idx], ref_clin.loc[idx]

mask = (ref_clin["uberon_tissue"].notna() &
        ~ref_clin["uberon_tissue"].isin(["unknown", "other", "unmapped", "nan", ""]))
ref_emb, ref_clin = ref_emb[mask], ref_clin[mask]

knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
knn.fit(ref_emb.values, ref_clin["uberon_tissue"].values)

knn_labels = knn.predict(embeddings)
print(knn_labels)
```

---

## 5. Interactive option

For a no-code interface, the same model runs as a web app:

streamlit run webapp/app.py

or try the hosted demo: https://huggingface.co/spaces/akalinLab/flexynesis-tissue-vae — a lightweight version for quick exploration. Because the full model weights exceed the free Hugging Face tier, the hosted demo approximates embeddings via TruncatedSVD and classifies by kNN, so its predictions are approximate. For exact VAE inference and confidence scores, run the script above with the model from Zenodo.
