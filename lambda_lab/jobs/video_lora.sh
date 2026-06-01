#!/usr/bin/env bash
# Train a video LoRA (Wan / HunyuanVideo / LTX-Video) with diffusion-pipe.
# Arg: path to the TOML config. Uses DeepSpeed on a single GPU; the example
# config enables block-swap so 14B models fit on 80 GB (and largely on 48 GB).
set -euo pipefail
source "$HOME/venv/bin/activate"
CONFIG="${1:-$HOME/lambda_lab/run.toml}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
cd "$HOME/diffusion-pipe"
echo "video LoRA training: $CONFIG"
deepspeed --num_gpus=1 train.py --deepspeed --config "$CONFIG"
echo "training finished -> $HOME/lambda_lab/output"
