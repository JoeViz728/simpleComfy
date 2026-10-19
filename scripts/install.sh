#!/usr/bin/env bash
# Install simpleComfy API (Python venv). ComfyUI is separate: install-comfyui.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "==> Creating venv"
uv venv .venv --python 3.12 || uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing simpleComfy"
uv pip install -e ".[dev]"

if [[ ! -f config.toml ]]; then
  cp config.example.toml config.toml
  echo "==> Wrote config.toml from example (edit comfy.root + tokens via env)"
fi

mkdir -p data/models/{checkpoints,loras,vae,embeddings,controlnet,upscale_models}
mkdir -p data/outputs data/state

echo ""
echo "simpleComfy installed."
echo "  Next:  ./scripts/install-comfyui.sh   # once"
echo "         export CIVITAI_API_TOKEN=…    # for pulls"
echo "         ./scripts/start.sh"
echo "  UI:    http://127.0.0.1:8790/"
echo "  API:   http://127.0.0.1:8790/v1/help"
