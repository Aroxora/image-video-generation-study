#!/usr/bin/env bash
# Start ComfyUI bound to localhost (reach it through the SSH tunnel the
# orchestrator prints — never expose 8188 to the public internet).
set -euo pipefail
source "$HOME/venv/bin/activate"
cd "$HOME/ComfyUI"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "ComfyUI starting on 127.0.0.1:8188"
python main.py --listen 127.0.0.1 --port 8188
