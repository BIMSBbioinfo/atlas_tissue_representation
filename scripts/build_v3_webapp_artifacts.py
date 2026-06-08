#!/usr/bin/env python3
"""v3 webapp artifacts generator — FIXED for Flexynesis attribute naming."""
import os, sys, glob, shutil, gc
import joblib, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from h5_dataloader import H5DataImporter
from train_denoising_vae import DenoisingVAE

RESULTS_DIR = 'results_denoising_vae_411k_B'
DATA_PATH   = 'processed_scaled_411k_tissue_B_h5'
STD_PTH     = f'{RESULTS_DIR}/standard_vae.pth'
STD_CKPT    = f'{RESULTS_DIR}/ckpt_standard/best_epoch=*.ckpt'
OUT_DIR = '/home/amit/Desktop/flexynesis_manuscript_final/webapp_v3/model'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[1] Loading v3 Standard VAE → {device}")
model = torch.load(STD_PTH, map_location='cpu', weights_only=False).to(device)
best_ckpt = sorted(glob.glob(STD_CKPT),
    key=lambda p: float(p.split('val_loss=')[1].split('.ckpt')[0]))[0]
print(f"    Best checkpoint: {os.path.basename(best_ckpt)}")
ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

print(f"\n[2] Loading data from {DATA_PATH}")
di = H5DataImporter(path=DATA_PATH, data_types=['gex'],
                    log_transform=False, top_percentile=100, min_features=100)
train_ds, test_ds = di.import_data()
print(f"    Train: {len(train_ds)}  Test: {len(test_ds)}")

# CORRECT Flexynesis attribute names (verified via diagnostic):
gene_list  = list(train_ds.features['gex'])
scaler     = di.scalers['gex']
label_enc  = di.label_encoders['uberon_tissue']

# label_mapping — prefer Flexynesis-provided, fallback to encoder categories
if 'uberon_tissue' in train_ds.label_mappings:
    raw = train_ds.label_mappings['uberon_tissue']
    label_mapping = {}
    for k, v in raw.items():
        try:    label_mapping[int(k)] = str(v)
        except: label_mapping[k] = str(v)
else:
    label_mapping = {i: str(c) for i, c in enumerate(label_enc.categories_[0])}

print(f"    Genes: {len(gene_list)}")
print(f"    Classes: {len(label_mapping)}")
print(f"    First 5 tissues: {list(label_mapping.values())[:5]}")
print(f"    Last 3 entries: {list(label_mapping.items())[-3:]}")

artifacts = {
    'feature_lists':  {'gex': gene_list},
    'transforms':     {'gex': scaler},
    'label_encoders': {'uberon_tissue': label_enc},
    'label_mapping':  label_mapping,
}
art_path = os.path.join(OUT_DIR, 'vae_tissue.artifacts.joblib')
joblib.dump(artifacts, art_path)
print(f"\n[3] Saved: {art_path}")
_ = joblib.load(art_path)  # round-trip check
print(f"    Round-trip OK")

def encode_split(ds, name, bs=256):
    print(f"\n[4-{name}] Encoding {len(ds)} samples...")
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    all_mu, all_samples = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            dat, _, samples = batch
            x  = dat['gex'].to(device)
            h  = model.encoders[0](x)
            mu = model.FC_mean(h[0])
            all_mu.append(mu.cpu().numpy())
            all_samples.extend(samples)
            if i % 20 == 0:
                print(f"    batch {i}/{len(loader)}", flush=True)
    emb = np.concatenate(all_mu)
    df = pd.DataFrame(emb, index=all_samples,
                      columns=[f'E{j+1}' for j in range(emb.shape[1])])
    out = os.path.join(OUT_DIR, f'embeddings_{name}.csv')
    df.to_csv(out)
    print(f"    Saved: {out}  shape={emb.shape}")

encode_split(train_ds, 'train')
encode_split(test_ds,  'test')

del model; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\n[5] Copying clinical CSVs + model checkpoint...")
shutil.copy(f'{DATA_PATH}/train/clin.csv', os.path.join(OUT_DIR, 'train_clin.csv'))
shutil.copy(f'{DATA_PATH}/test/clin.csv',  os.path.join(OUT_DIR, 'test_clin.csv'))
print(f"    Clinical CSVs copied")

model_dest = os.path.join(OUT_DIR, 'vae_tissue.final_model.pth')
if os.path.exists(model_dest) or os.path.islink(model_dest):
    os.remove(model_dest)
os.symlink(os.path.abspath(STD_PTH), model_dest)
print(f"    Model symlinked (for HF: replace with: cp {STD_PTH} {model_dest})")

print(f"\n{'='*65}\nv3 WEBAPP ARTIFACTS — READY\n{'='*65}")
print(f"Output directory: {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, f)
    real = os.path.realpath(fp) if os.path.islink(fp) else fp
    sz = os.path.getsize(real)
    marker = " (symlink)" if os.path.islink(fp) else ""
    print(f"  {f}{marker}  ({sz/1e6:,.1f} MB)")
