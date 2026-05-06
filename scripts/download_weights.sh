#!/usr/bin/env bash
set -euo pipefail

# Downloads Zyphra/ZAYA1-8B weights to the local Hugging Face cache.
# Re-runnable: HF cache dedupes by content hash so reruns are fast.
#
# Uses the huggingface_hub Python API rather than the `hf` CLI because the CLI
# (as of huggingface_hub 0.36.x) misinterprets --include glob patterns when
# called from a shell script.

REPO="Zyphra/ZAYA1-8B"
PY_BIN="$(dirname "$0")/../reference/.venv/bin/python"

if [ ! -x "$PY_BIN" ]; then
  echo "Reference venv not found at $PY_BIN. Run uv venv setup first." >&2
  exit 1
fi

echo "Downloading $REPO to local HF cache..."
"$PY_BIN" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Zyphra/ZAYA1-8B",
    allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
)
print(f"Snapshot path: {path}")
PY

echo
echo "Cache size for $REPO:"
du -sh ~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B 2>/dev/null || true
