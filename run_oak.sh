#!/usr/bin/env bash
# Run the vision pipeline with OAK-D using sudo (fixes USB permission on some macOS setups).
# Run from a shell where your conda env is active so the same Python (with numpy, depthai) is used.
# Usage: ./run_oak.sh [extra args...]
# Example: ./run_oak.sh --vlm --max-frames 100
set -e
cd "$(dirname "$0")"
# Use the exact Python currently in use (same interpreter that has numpy, depthai, etc.)
PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
exec sudo env PATH="${PATH}" HOME="${HOME}" "${PYTHON}" run_vision_pipeline.py --backend oak "$@"
