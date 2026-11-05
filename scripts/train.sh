#!/usr/bin/env bash
# Usage: METHOD=grpo|rltt|lrpg [LAM=0] [FIT_ALPHA=1] [ALPHA=1.0] [DIV=js|kl] [SETUP=math|sdpo] [TASK=chemistry]
#        [WEIGHTS=uniform] [RUN=name] [GPUS=0,1] scripts/train.sh [extra hydra overrides]
set -euo pipefail
export LOOPRL_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source /venv/rl/bin/activate

METHOD=${METHOD:-grpo}
LAM=${LAM:-0}            # deep-supervision weight; > 0 enables it
FIT_ALPHA=${FIT_ALPHA:-1}  # lrpg: refit alpha every batch (1) or keep ALPHA fixed (0)
ALPHA=${ALPHA:-1.0}
DIV=${DIV:-js}           # lrpg: loop divergence, js | kl
SETUP=${SETUP:-math}
TASK=${TASK:-chemistry}  # sdpo only: biology | chemistry | material | physics | tooluse
WEIGHTS=${WEIGHTS:-uniform}
GPUS=${GPUS:-0,1}
NAME=$METHOD
if [ "$METHOD" = lrpg ] && [ "$DIV" != js ]; then NAME+=-$DIV; fi
if [ "$METHOD" = lrpg ] && [ "$FIT_ALPHA" = 0 ]; then NAME+=-a$ALPHA; fi
if [ "$LAM" != 0 ]; then NAME+=-ds$LAM; fi
case $SETUP in
  math) RUN=${RUN:-$NAME}; SETUP_ARGS="setup=math" ;;
  sdpo) RUN=${RUN:-$NAME-$TASK}; SETUP_ARGS="setup=sdpo task=$TASK" ;;
  *) echo "unknown SETUP=$SETUP"; exit 1 ;;
esac
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

export CUDA_VISIBLE_DEVICES=$GPUS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export PYTHONWARNINGS=ignore
export WANDB_DIR=/dev/shm/looprl/wandb
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}
export RAY_TMPDIR=/dev/shm/looprl/ray/$RUN
mkdir -p "$LOOPRL_ROOT/logs" "$WANDB_DIR" "$RAY_TMPDIR" /dev/shm/looprl/ckpt

# known-harmless chatter dropped from the console/.log stream
NOISE='is not available: ModuleNotFoundError|warnings\.warn\(|incorrect regex pattern|Continuous Token builder'
NOISE+='|blocking ray\.get inside async actor|SimpleStorageUnit|Time range: last|jit_monitor|flashinfer\.py'
NOISE+='|OMP_NUM_THREADS|Detected \[openai\]|W&B Weave|weave-docs|Loading weights: +[0-9]+%'

LOG="$LOOPRL_ROOT/logs/train_$RUN.log"
echo "[$(date '+%F %T')] launch METHOD=$METHOD LAM=$LAM FIT_ALPHA=$FIT_ALPHA ALPHA=$ALPHA DIV=$DIV SETUP=$SETUP_ARGS WEIGHTS=$WEIGHTS GPUS=$GPUS RUN=$RUN $*" | tee -a "$LOG"
cd "$LOOPRL_ROOT"
python -m looprl.main $SETUP_ARGS \
  looprl.method="$METHOD" looprl.alpha="$ALPHA" looprl.fit_alpha="$([ "$FIT_ALPHA" = 0 ] && echo false || echo true)" \
  looprl.divergence="$DIV" looprl.weights="$WEIGHTS" looprl.lam="$LAM" \
  trainer.experiment_name="$RUN" trainer.n_gpus_per_node="$N_GPUS" \
  "$@" 2>&1 | sed -u 's/\x1b\[[0-9;]*m//g' | grep --line-buffered -vE "$NOISE" | tee -a "$LOG"
