#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

"$PYTHON_BIN" "$ROOT/pipeline/compute_features.py" \
  --data_dir "$ROOT/data" \
  --protbert_path "$ROOT/models/protbert_finetuned" \
  --output_dir "$ROOT/features" \
  --device "$DEVICE"

"$PYTHON_BIN" "$ROOT/pipeline/train_token_attn.py" \
  --feature_dir "$ROOT/features" \
  --output_dir "$ROOT/models/token_attention" \
  --device "$DEVICE"
