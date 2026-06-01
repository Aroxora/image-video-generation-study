#!/usr/bin/env bash
# Train a FLUX LoRA with Ostris ai-toolkit. Arg: path to the YAML config.
# The example config writes checkpoints to ~/lambda_lab/output, which the
# orchestrator rsyncs back before teardown.
set -euo pipefail
source "$HOME/venv/bin/activate"
CONFIG="${1:-$HOME/lambda_lab/run.yaml}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# FLUX.1-dev is gated — `huggingface-cli login` (or set HF_TOKEN) once if needed.
cd "$HOME/ai-toolkit"
echo "FLUX LoRA training: $CONFIG"
python run.py "$CONFIG"
echo "training finished -> $HOME/lambda_lab/output"
