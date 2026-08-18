#!/bin/bash
# Stop cleanly before midnight rather than trusting an estimate. At the soft
# deadline we stop LAUNCHING work (the pause file the queues already honour) and
# let what is running finish; at the hard deadline we stop the director. Nothing
# is lost either way: every job writes to <out>.part and is renamed only on
# success, so an interrupted job simply re-runs later with --skip-existing.
set -u
source /raid/home/CAMCA/hj880/medvigil_env.sh
cd $MEDVIGIL3D_ROOT
PAUSE=$MEDVIGIL3D_ROOT/runs/.exclusive.pause
SOFT=$(date -d "today 23:40" +%s); HARD=$(date -d "today 23:58" +%s)

while [ "$(date +%s)" -lt "$SOFT" ]; do sleep 60; done
echo "[$(date +%T)] soft deadline: no new jobs will start"
touch "$PAUSE"

while [ "$(date +%s)" -lt "$HARD" ]; do
  n=$(pgrep -u "$(whoami)" -fc "run_identification_contro|run_subtask|run_inference_comput|run_input_richness|run_leakage|run_roi_arms|sanity_controls|run_multimodel" 2>/dev/null || echo 0)
  [ "$n" -le 1 ] && break
  sleep 30
done
echo "[$(date +%T)] stopping the director"
screen -S director -X quit 2>/dev/null
for p in $(pgrep -u "$(whoami)" -f "runs/run_queu[e].py"); do kill $p 2>/dev/null; done
sleep 5
rm -f "$PAUSE" $MEDVIGIL3D_ROOT/results_new/*.part $MEDVIGIL3D_ROOT/logs/*.claim
echo "[$(date +%T)] STOPPED. completed outputs are intact; re-run with --skip-existing to continue"
python runs/summarise.py > $MEDVIGIL3D_ROOT/results_new/FINAL_SUMMARY.txt 2>&1
echo "[$(date +%T)] wrote results_new/FINAL_SUMMARY.txt"
