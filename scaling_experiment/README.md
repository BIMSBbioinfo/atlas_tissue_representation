# Scaling experiment: performance vs. training-set size

How test performance of the supervised VAE changes with the number of training samples.

**How the stats were obtained**
- Data: 118,263 train / ~28K test bulk RNA-seq samples, 16,115 genes (`gex`), target `uberon_tissue`.
- Model: flexynesis 1.1.14 `supervised_vae`, one HPO iteration (`--hpo_iter 1`) with hyperparameters pinned in `config.yaml`
  (latent_dim 58-59, hidden_dim_factor 0.5, supervisor_hidden_dim 61-62, lr 1e-4, **batch size 1024**, max 200 epochs, early stopping with patience 10).
  Batch size 1024 (instead of 256 in the full-data reference model) was chosen for speed.
- Training-set sizes: N = 11,800 ... 118,000 in steps of 11,800 (10% ... 100% of the training set), drawn with `--subsample N`
  (random, not stratified by tissue; rare tissues can be thin or absent at small N). `--features_top_percentile 100` keeps all
  features at every N. The test set is always the full held-out test split.
- Replicates: 10 seeds (`--seed 1..10`) per N. The subsample and the training both change with the seed.
- Each run is launched by `run_one.sh N SEED`; only the small per-run outputs are kept in `stats/`:
  `nN_sS.stats.csv` (test metrics), `.final_model_config.json` (incl. epochs trained), `.timing.txt` (wall-clock).

**Files**
- `make_scaling_figure.py`: aggregates `stats/` and draws the figure (`python make_scaling_figure.py`).
- `scaling_results_all_runs.csv`, `scaling_results_summary.csv`: per-run and mean/SD per N.
- `scaling_figure.png`, legend in `LEGEND.md`.
