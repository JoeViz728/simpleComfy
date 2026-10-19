#!/usr/bin/env bash
# Start simpleComfy API. Optionally starts ComfyUI if comfy.root is set and not reachable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "No .venv — run ./scripts/install.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ ! -f config.toml ]]; then
  cp config.example.toml config.toml
fi

export SIMPLECOMFY_CONFIG="${SIMPLECOMFY_CONFIG:-$ROOT/config.toml}"

HOST="${SIMPLECOMFY_HOST:-127.0.0.1}"
PORT="${SIMPLECOMFY_PORT:-8790}"

echo "==> simpleComfy http://${HOST}:${PORT}"
echo "    config: $SIMPLECOMFY_CONFIG"
if [[ -n "${CIVITAI_API_TOKEN:-}" ]]; then
  echo "    CIVITAI_API_TOKEN: set"
else
  echo "    CIVITAI_API_TOKEN: (not set — pulls may fail for gated files)"
fi

exec python -m simplecomfy serve --host "$HOST" --port "$PORT"
