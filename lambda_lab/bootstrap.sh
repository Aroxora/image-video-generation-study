#!/usr/bin/env bash
# Remote installer, run once per instance by the bootstrap step.
#   bootstrap.sh <stack>   where stack = diffusers | comfyui | ai-toolkit | diffusion-pipe
#
# It is arch-aware: the cheapest big-VRAM box (GH200, $2.29/hr, 96 GB) is ARM64
# (aarch64 / sbsa), the rest are x86_64. PyTorch ships CUDA wheels for BOTH from
# the same index, so the venv install is identical; what differs on ARM is that a
# few extras (flash-attention, some prebuilt kernels) may need source builds, so
# we install conservative defaults and let each tool pull what it needs.
#
# Idempotent: re-running reuses the venv and skips existing clones.
set -euo pipefail

STACK="${1:-diffusers}"
ARCH="$(uname -m)"                 # aarch64 (GH200) | x86_64
KIT="$HOME/lambda_lab"

echo "== lambda_lab bootstrap: stack=$STACK arch=$ARCH =="

# Find a persistent filesystem mount (survives teardown), if one is attached, and
# keep the venv + HuggingFace cache there so scale-to-zero restarts skip the
# multi-GB reinstall/download. Falls back to the ephemeral instance SSD.
FS=""
for cand in /home/ubuntu/*/ /lambda/*/; do
  [ -d "$cand" ] && [ -w "$cand" ] && [ "$cand" != "$HOME/" ] && { FS="${cand%/}"; break; }
done
if [ -n "$FS" ]; then
  VENV="$FS/venv"; export HF_HOME="$FS/hf-cache"
  ln -sfn "$VENV" "$HOME/venv"      # so $HOME/venv/bin/python always resolves
  echo "persistent fs: $FS — venv + weights cached here across restarts"
else
  VENV="$HOME/venv"; export HF_HOME="$HOME/.cache/huggingface"
  echo "no persistent fs — using ephemeral SSD (cold starts re-download weights)"
fi
mkdir -p "$HF_HOME"
grep -q "export HF_HOME=$HF_HOME" "$HOME/.bashrc" 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> "$HOME/.bashrc"
echo "HF cache -> $HF_HOME"

sudo apt-get update -y -qq || true
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-venv python3-pip git git-lfs ffmpeg aria2 >/dev/null 2>&1 || true
git lfs install || true

# --- python venv + CUDA PyTorch -------------------------------------------
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel setuptools

# CUDA 12.4 wheels exist for x86_64 AND aarch64 (sbsa). Lambda boxes ship a
# recent driver; if torch is already importable with CUDA we skip the reinstall.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "installing torch (cu124, $ARCH) …"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))"

common=(diffusers transformers accelerate safetensors sentencepiece protobuf "huggingface_hub[cli]" pillow)

case "$STACK" in
  diffusers)
    pip install -q "${common[@]}"
    ;;
  comfyui)
    [ -d "$HOME/ComfyUI" ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$HOME/ComfyUI"
    pip install -q -r "$HOME/ComfyUI/requirements.txt"
    ;;
  ai-toolkit)
    # Ostris ai-toolkit — the most beginner-friendly FLUX / SDXL LoRA trainer.
    [ -d "$HOME/ai-toolkit" ] || git clone --depth 1 https://github.com/ostris/ai-toolkit "$HOME/ai-toolkit"
    pip install -q "${common[@]}" peft optimum-quanto prodigyopt lycoris-lora
    pip install -q -r "$HOME/ai-toolkit/requirements.txt" || true
    ;;
  diffusion-pipe)
    # tdrussell/diffusion-pipe — video LoRA (Wan / HunyuanVideo / LTX) via DeepSpeed.
    [ -d "$HOME/diffusion-pipe" ] || git clone --depth 1 --recurse-submodules https://github.com/tdrussell/diffusion-pipe "$HOME/diffusion-pipe"
    pip install -q "${common[@]}" deepspeed peft toml
    pip install -q -r "$HOME/diffusion-pipe/requirements.txt" || true
    ;;
  *)
    echo "unknown stack: $STACK" >&2; exit 2 ;;
esac

# nice-to-have, best-effort (may build from source on aarch64 — never fatal)
pip install -q bitsandbytes 2>/dev/null || echo "(bitsandbytes skipped on $ARCH — fine)"

echo "== bootstrap done (stack=$STACK) =="
