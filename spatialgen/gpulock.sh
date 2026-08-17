# Shared GPU arbitration for the experiment queues.
#
# Why this exists
# ---------------
# The queues previously each carried their own "wait until a card looks free"
# loop. Two loops polling the same two cards is not mutual exclusion: one queue
# reads the free memory a moment before the other allocates, both conclude the
# card is available, and both land on it. That happened -- a 32B ROI run and a
# LLaVA-OneVision run shared cuda:0, and OneVision, which allocates ~4 GiB per
# multi-tile montage, took 2185 CUDA OOMs and wrote 77 of 2262 probes.
#
# flock makes the check-and-claim atomic: the lock is held for the whole
# lifetime of the python process, not just the memory read, so a second queue
# cannot observe the card between the decision and the allocation.
#
# The second half of the file addresses how that failure stayed invisible. The
# queue's resume guard was `[ -s "$OUT" ]` -- true for a 77-row file -- so the
# truncated output would have been treated as complete on the next pass and
# folded into the tables. A run is now complete only if it has the number of
# rows it was asked for.
LOCKDIR=/tmp/claude-1007/-home-hanqijiang/gpulocks
mkdir -p "$LOCKDIR"
GPU_TOTAL_MIB=95830

gpu_acquire () {  # need_mib -- sets $GPU, holds the lock until gpu_release
  local need=$1 i used
  while true; do
    for i in 0 1; do
      exec {LOCKFD}>"$LOCKDIR/gpu$i.lock"
      if flock -n "$LOCKFD"; then
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        # free memory is checked *while holding the lock*, so an unlocked
        # process (an older queue still finishing) is still respected
        if [ $((GPU_TOTAL_MIB - used)) -gt "$need" ]; then GPU="cuda:$i"; return 0; fi
      fi
      exec {LOCKFD}>&-
    done
    sleep 120
  done
}

gpu_release () { exec {LOCKFD}>&- 2>/dev/null; }

complete () {  # file expected_rows -- is this output finished?
  [ -s "$1" ] && [ "$(wc -l < "$1")" -eq "$2" ]
}

# Run a command holding a card, then verify the output before accepting it.
# A short file is deleted rather than left on disk, so a resume re-runs it
# instead of skipping it.
gpu_run () {  # need_mib out expected_rows label cmd...
  local need=$1 out=$2 want=$3 label=$4; shift 4
  if complete "$out" "$want"; then echo "skip $label ($want rows present)"; return 0; fi
  [ -e "$out" ] && { echo "discard partial $label ($(wc -l < "$out")/$want)"; rm -f "$out"; }
  gpu_acquire "$need"
  echo "=== $label on $GPU start $(date -Is) ==="
  "$@" --device "$GPU" 2>&1 | grep -vE "^Loading|it/s\]|%\||Fetching|^Warning: You are sending"
  gpu_release
  local got=0; [ -e "$out" ] && got=$(wc -l < "$out")
  if [ "$got" -eq "$want" ]; then
    echo "=== $label OK $got rows $(date -Is) ==="
  else
    echo "=== $label INCOMPLETE $got/$want rows -- discarded $(date -Is) ==="
    rm -f "$out"; return 1
  fi
}
