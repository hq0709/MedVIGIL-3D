#!/bin/bash
# Qwen2.5-VL-72B: 146.8 GB of bf16 weights, so it needs both
# cards sharded and cannot share either with anything else.
# Run only when GPU 6 has no other tenant.
set -u
source /raid/home/CAMCA/hj880/medvigil_env.sh
cd $MEDVIGIL3D_ROOT
export CUDA_VISIBLE_DEVICES=6,7
for O in Task03_Liver Task06_Lung Task07_Pancreas Task10_Colon; do
  for C in plain identified; do
    OUT=results_new/id_${O}_qwen72b_${C}.jsonl
    [ -s "$OUT" ] && continue
    echo "[$(date +%T)] qwen72b $O $C"
    python spatialgen/run_identification_control.py --qa cfqa_$O/qa \
      --task-dir $MSD_ROOT/$O --seg-cache cfqa_$O/seg_cache \
      --model qwen72b --condition $C --subset matched \
      --device auto --out $OUT.part 2>&1 | tail -2
    [ -s "$OUT.part" ] && mv "$OUT.part" "$OUT"
  done
done
echo "[$(date +%T)] 72B IMAGE ARMS DONE"
