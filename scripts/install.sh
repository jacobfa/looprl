#!/usr/bin/env bash
# Builds the /venv/rl environment: torch 2.13 + vLLM 0.32 + flash-attn 2.8.3 + verl (pinned main) + looprl.
set -euo pipefail

VENV=${VENV:-/venv/rl}
VERL_COMMIT=${VERL_COMMIT:-31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

uv venv "$VENV" --python 3.12 --seed
source "$VENV/bin/activate"

uv pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install vllm==0.32.0 transformers==5.16.1 torch==2.13.0
uv pip install --no-deps \
  https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3-py3.12-torch2.13.0-cu130/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
uv pip install "verl[verl-core] @ git+https://github.com/volcengine/verl@${VERL_COMMIT}" \
  torch==2.13.0 vllm==0.32.0 transformers==5.16.1 cupy-cuda13x==14.0.1
uv pip install math-verify "huggingface_hub[cli,hf_transfer]" datasets wandb pyyaml
uv pip install -e "$ROOT"

python - <<'EOF'
import torch, vllm, transformers, verl, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("vllm", vllm.__version__, "transformers", transformers.__version__, "verl", verl.__version__, "flash_attn", flash_attn.__version__)
EOF
