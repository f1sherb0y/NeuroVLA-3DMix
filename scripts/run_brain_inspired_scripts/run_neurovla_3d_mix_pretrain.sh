#!/usr/bin/env bash
# Run the same LIBERO pre-training recipe as NeuroVLA with 3D-MIX enabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=neuro_vla_3d_mix exec bash "$SCRIPT_DIR/run_neurovla_pretrain.sh" "$@"
