#!/bin/bash
# E4: Qwen2.5-VL-72B needs 146.8 GB of bf16 weights sharded across BOTH cards,
# so it cannot share either one. GPU 6 currently has another user's job on it.
# Rather than skip the experiment, wait for that card to clear and take both
# cards at that moment, pausing the wave queues instead of killing them.
set -u
source /raid/home/CAMCA/hj880/medvigil_env.sh
cd $MEDVIGIL3D_ROOT
PAUSE=$MEDVIGIL3D_ROOT/runs/.exclusive.pause
ME=$(whoami)

foreign_on () {                     # foreign_on <gpu> -> prints GB held by others
  local uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1)
  local tot=0
  while IFS=, read -r u pid mem; do
    u=$(echo $u | xargs); pid=$(echo $pid | xargs); mem=$(echo $mem | xargs)
    [ "$u" = "$uuid" ] || continue
    local owner=$(ps -o user= -p $pid 2>/dev/null | xargs)
    [ -n "$owner" ] && [ "$owner" != "$ME" ] && tot=$((tot + mem))
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
  echo $tot
}

mine_running () {                   # my compute processes on 6 or 7
  local n=0
  for g in 6 7; do
    local uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g)
    while IFS=, read -r u pid mem; do
      u=$(echo $u | xargs); pid=$(echo $pid | xargs)
      [ "$u" = "$uuid" ] || continue
      local owner=$(ps -o user= -p $pid 2>/dev/null | xargs)
      [ "$owner" = "$ME" ] && n=$((n + 1))
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
  done
  echo $n
}

done_count () { ls results_new/id_Task*_qwen72b_{plain,identified}.jsonl 2>/dev/null | wc -l; }

echo "[$(date +%T)] watching for a free GPU 6; 72B needs both cards"
while [ "$(done_count)" -lt 8 ]; do
  f6=$(foreign_on 6); f7=$(foreign_on 7)
  if [ "$f6" -eq 0 ] && [ "$f7" -eq 0 ]; then
    echo "[$(date +%T)] both cards free of other users; claiming them"
    touch "$PAUSE"
    # let the wave queues finish what they already started, then take the cards
    for _ in $(seq 1 60); do
      [ "$(mine_running)" -eq 0 ] && break
      sleep 30
    done
    if [ "$(foreign_on 6)" -eq 0 ] && [ "$(foreign_on 7)" -eq 0 ]; then
      echo "[$(date +%T)] running the 72B image arms"
      bash runs/jobs_e4_72b.sh 2>&1 | tail -40
      echo "[$(date +%T)] 72B arms now at $(done_count)/8"
    else
      echo "[$(date +%T)] a co-tenant reappeared while draining; backing off"
    fi
    rm -f "$PAUSE"
  fi
  sleep 120
done
rm -f "$PAUSE"
echo "[$(date +%T)] E4 COMPLETE: $(done_count)/8"
