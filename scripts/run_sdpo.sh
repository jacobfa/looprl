#!/usr/bin/env bash
# SDPO protocol: every arm on every task, each for looprl.max_train_hours (5 h by default) of training time.
# ARMS entries are method[:lam], e.g. lrpg:1 = LRPG with deep supervision weight 1.
set -uo pipefail
cd "$(dirname "$0")/.."
for t in ${TASKS:-chemistry physics biology material tooluse}; do
  for arm in ${ARMS:-lrpg lrpg:1 rltt grpo}; do
    m=${arm%%:*}
    lam=$([[ $arm == *:* ]] && echo "${arm#*:}" || echo 0)
    echo "[$(date '+%F %T')] start $arm $t" | tee -a logs/run_sdpo.log
    SETUP=sdpo TASK=$t METHOD=$m LAM=$lam scripts/train.sh "$@"
    echo "[$(date '+%F %T')] end $arm $t (exit $?)" | tee -a logs/run_sdpo.log
  done
done
