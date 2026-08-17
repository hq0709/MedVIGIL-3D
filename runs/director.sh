#!/bin/bash
# Run the remaining programme wave by wave. Both cards work the same job list and
# partition it by atomic claim; the GPU-6 budget is recomputed at every wave from
# who else is on the card, so a co-tenant arriving mid-programme shrinks our
# share instead of OOM-ing them. Queues honour an exclusive-pause file so the
# 72B, which needs both cards sharded, can take them without losing their place.
set -u
source /raid/home/CAMCA/hj880/medvigil_env.sh
cd $MEDVIGIL3D_ROOT
L=/raid/home/CAMCA/hj880/logs
PAUSE=$MEDVIGIL3D_ROOT/runs/.exclusive.pause

wave () {
  local name=$1 jobs=$2
  [ -s "$jobs" ] || { echo "[$(date +%T)] $name: no job file"; return; }
  echo "[$(date +%T)] === wave $name ($(wc -l < $jobs) jobs)"
  rm -f logs/*.claim
  python runs/run_queue.py --jobs "$jobs" --gpus 7 --share-headroom 50 \
      --pause-file "$PAUSE" --skip-existing > $L/wave_${name}_gpu7.log 2>&1 &
  local p7=$!
  sleep 8
  python runs/run_queue.py --jobs "$jobs" --gpus 6 --share-headroom 50 \
      --pause-file "$PAUSE" --skip-existing > $L/wave_${name}_gpu6.log 2>&1 &
  local p6=$!
  wait $p7 $p6
  echo "[$(date +%T)] --- wave $name done: $(grep -hcE '^\[ok ' $L/wave_${name}_gpu*.log | paste -sd+ | bc) ok, $(grep -hcE '^\[FAIL' $L/wave_${name}_gpu*.log | paste -sd+ | bc) failed"
}

while screen -ls 2>/dev/null | grep -qE "\.q[67]\b"; do sleep 60; done
echo "[$(date +%T)] E1 re-run finished: $(ls results_new/id_Task*_{internvl,qwen3vl,qwen7b,qwen32b}_{plain,bestslice,overlay,identified}.jsonl 2>/dev/null | wc -l)/64"

wave e1       runs/jobs_e1_all.jsonl
wave e3e5text runs/jobs_e3e5.jsonl
wave e9       runs/jobs_e9.jsonl
wave e12      runs/jobs_e12.jsonl
wave e6       runs/jobs_e6.jsonl
wave e5       runs/jobs_e5.jsonl
wave e10      runs/jobs_e10.jsonl
wave e8       runs/jobs_e8.jsonl
wave e11      runs/jobs_e11.jsonl
echo "[$(date +%T)] PROGRAMME COMPLETE"
