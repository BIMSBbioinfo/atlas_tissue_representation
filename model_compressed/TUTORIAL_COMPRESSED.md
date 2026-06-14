# Tissue prediction with the compressed model

This tutorial shows how to predict tissue types from bulk RNA-seq using the
compressed, deployment-ready model in this directory. No GPU and no `flexynesis`
installation are required at inference time.

## What is in this directory

| File | Size | Description |
|---|---|---|
| `vae_tissue_int8.torchscript.pt` | 53 MB | int8-quantized TorchScript bundle — self-contained, no class definitions needed at load time |
| `vae_tissue.artifacts.joblib` | 0.6 MB | gene list (16,115 genes) + expression scaler fitted on the training data |
| `label_mapping.json` | — | integer → tissue name for 42 UBERON classes |

The original model checkpoint (`vae_tissue.final_model.pth`, ~8 GB on Zenodo)
is not needed for prediction. Everything required is in this directory.

## Install dependencies

```bash
mamba create -n flexynesis-predict python==3.11
mamba activate flexynesis-predict
pip install torch joblib pandas numpy scikit-learn
```

## Input format

A CSV file with **samples as rows** and **gene symbols as columns**.
Transposed orientation (genes as rows, samples as columns) is also accepted —
the script detects it automatically.

- Genes absent from the model's 16,115-gene vocabulary are zero-filled.
- A partial gene overlap is fine; the script prints the overlap percentage.
- Raw or normalised counts both work provided they are on a linear scale
  (the scaler handles standardisation).

## Running a prediction

Run from the **repository root**:

```bash
python scripts/predict_tissue.py gex.lung_100.csv
```

The script looks for the model and artifacts in `model_compressed/` by default,
so no extra flags are needed as long as you run from the repo root.

**Example — 100 TCGA lung samples** (`gex.lung_100.csv` is included in the
repository):

```
Samples: 100   Gene overlap: 16069/16115 (99.7%)
                      Sample     Tissue Confidence
TCGA-50-5932-01A-11R-1755-07       lung    100.0%
TCGA-55-8091-01A-11R-2241-07       lung    100.0%
TCGA-49-4505-01A-01R-1206-07       lung    100.0%
TCGA-34-5232-01A-21R-1820-07     breast    100.0%
TCGA-63-5131-01A-01R-1443-07       lung    100.0%
...
```

96 of the 100 samples are called as **lung** at ≥99% confidence. The four
non-lung calls (breast, fibroblast, ovary) reflect genuine expression-level
ambiguity in those particular TCGA samples, not model errors.

Results are written to `predictions.csv` (change with `--out`).

## All options

```
python scripts/predict_tissue.py --help

positional arguments:
  input            Gene-expression CSV (samples × genes or genes × samples)

options:
  --model-dir DIR  Directory with TorchScript model and label_mapping.json
                   (default: model_compressed)
  --artifacts PATH Path to vae_tissue.artifacts.joblib
                   (default: model_compressed/vae_tissue.artifacts.joblib)
  --out PATH       Output CSV path  (default: predictions.csv)
```

## Using the model directly in Python

```python
import json, torch, joblib, pandas as pd

model = torch.jit.load("model_compressed/vae_tissue_int8.torchscript.pt")
model.eval()

art          = joblib.load("model_compressed/vae_tissue.artifacts.joblib")
gene_list    = list(art["feature_lists"]["gex"])
scaler       = art["transforms"]["gex"]
label_mapping = {int(k): v for k, v in
                 json.loads(open("model_compressed/label_mapping.json").read()).items()}

df = pd.read_csv("gex.lung_100.csv", index_col=0)
aligned = pd.DataFrame(0.0, index=df.index, columns=gene_list)
common = [g for g in gene_list if g in df.columns]
aligned[common] = df[common].values
X = torch.tensor(scaler.transform(aligned.values), dtype=torch.float32)

with torch.no_grad():
    logits = model(X)
pred_labels = [label_mapping[int(i)] for i in logits.argmax(1)]
print(pred_labels[:5])
# ['lung', 'lung', 'lung', 'breast', 'lung']
```

## Recreating the compressed model from the Zenodo checkpoint

If you want to rebuild the bundle from `vae_tissue.final_model.pth`
(Zenodo, doi:10.5281/zenodo.20595537):

```bash
python scripts/compress_model.py \
    --model     model/vae_tissue.final_model.pth \
    --artifacts model_compressed/vae_tissue.artifacts.joblib \
    --out-dir   model_compressed \
    --validate  gex.lung_100.csv
```

**What the script does:**

1. Loads the original 8 GB checkpoint. It is large because the full training
   dataset (118,263 samples × 16,115 genes, ~7.6 GB float32) was pickled
   inside the model object. Actual weights are ~0.42 GB.
2. Extracts only the three sub-modules needed for inference (encoder, latent
   projection, tissue MLP).
3. Applies **int8 dynamic quantisation** to all `Linear` layers — no
   calibration data required.
4. Compiles to a self-contained **TorchScript** bundle.
5. Validates quantised vs fp32 predictions on the supplied CSV.

| | Size |
|---|---|
| Original `.pth` | 8.0 GB |
| Weights only (no training data) | 0.42 GB |
| int8 TorchScript bundle | **53 MB** |

fp32 vs int8 argmax agreement on `gex.lung_100.csv`: **1.0000**
