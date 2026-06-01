#!/usr/bin/env bash
# Train an SDXL LoRA with Ostris ai-toolkit. Arg: path to the YAML config.
# SDXL LoRA fits comfortably in 24 GB, so this is the A10 ($1.29/hr) job.
set -euo pipefail
source "$HOME/venv/bin/activate"
CONFIG="${1:-$HOME/lambda_lab/run.yaml}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
cd "$HOME/ai-toolkit"
echo "SDXL LoRA training: $CONFIG"
python run.py "$CONFIG"
echo "training finished -> $HOME/lambda_lab/output"
