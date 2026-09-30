#!/usr/bin/env bash
# ============================================================================
# Fetch or generate .onnx model weights for the inference engine.
# In production, replace this with curl/wget/git-lfs links to trained model weights.
# ============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo ">> Checking ONNX models in: $DIR"

# If models do not exist, run generator script
if [ ! -f "$DIR/task1_universal_ae.onnx" ]; then
    echo ">> Model files not found. Generating baseline ONNX models..."
    python3 "$DIR/generate_initial_models.py"
else
    echo ">> All ONNX model files are present."
fi

