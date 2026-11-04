#!/usr/bin/env bash
# Runs the three arms back to back on the same GPUs; each arm resumes from its last checkpoint if restarted.
set -uo pipefail
cd "$(dirname "$0")/.."
for m in ${METHODS:-lrpg rltt grpo}; do
  echo "[$(date '+%F %T')] start $m" | tee -a logs/run_all.log
  METHOD=$m scripts/train.sh "$@"
  echo "[$(date '+%F %T')] end $m (exit $?)" | tee -a logs/run_all.log
done
