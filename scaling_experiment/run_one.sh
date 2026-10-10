#!/bin/bash
# usage: run_one.sh N SEED
# Keeps only <prefix>.stats.csv, <prefix>.final_model_config.json and timing.txt per run; deletes the rest.
source /home/buyar/miniforge3/etc/profile.d/conda.sh
conda activate flexynesis
cd /home/buyar/test_vae_118K/scaling_experiment
N=$1
SEED=$2
P=n${N}_s${SEED}
OUT=runs/$P
mkdir -p $OUT logs
START=$(date +%s)
flexynesis \
  --data_path /home/buyar/test_vae_118K/data/processed_scaled_411k_tissue_B_h5 \
  --model_class supervised_vae \
  --data_types gex \
  --target_variables uberon_tissue \
  --config_path config.yaml \
  --hpo_iter 1 \
  --subsample $N \
  --features_top_percentile 100 \
  --disable_marker_finding \
  --seed $SEED \
  --device cuda \
  --outdir $OUT \
  --prefix $P > logs/$P.log 2>&1
RC=$?
echo "N=$N seed=$SEED exit=$RC wallclock_s=$(( $(date +%s) - START ))" > $OUT/timing.txt
# delete everything except the small files we need
find $OUT -type f ! -name "$P.stats.csv" ! -name "$P.final_model_config.json" ! -name timing.txt -delete
