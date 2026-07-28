#!/usr/bin/env bash
# Build one Conda environment for NeuroVLA training and in-process LIBERO eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

ENV_NAME="${CONDA_ENV:-neurovla}"
LIBERO_HOME="${LIBERO_HOME:-$PROJECT_ROOT/third_party/LIBERO}"
LIBERO_REPOSITORY="${LIBERO_REPOSITORY:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
LIBERO_REVISION="${LIBERO_REVISION:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
MUJOCO_GL_BACKEND="${MUJOCO_GL:-egl}"
SKIP_ENV_UPDATE=false
RUN_SIMULATOR_SMOKE=false

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_neurovla_env.sh [options]

Options:
  --env-name NAME        Conda environment name (default: neurovla)
  --libero-home PATH     LIBERO checkout (default: third_party/LIBERO)
  --skip-env-update      Keep the existing Conda packages
  --simulator-smoke      Create and step one headless LIBERO simulator
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --libero-home) LIBERO_HOME="$2"; shift 2 ;;
        --skip-env-update) SKIP_ENV_UPDATE=true; shift ;;
        --simulator-smoke) RUN_SIMULATOR_SMOKE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Non-interactive server shells often have Conda installed but not initialized.
if ! command -v conda >/dev/null 2>&1; then
    for CONDA_SH in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh"
    do
        if [ -f "$CONDA_SH" ]; then
            # shellcheck disable=SC1090
            source "$CONDA_SH"
            break
        fi
    done
fi

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is not available; install Miniconda or initialize it in this shell" >&2
    exit 1
}
command -v git >/dev/null 2>&1 || { echo "ERROR: git is not available" >&2; exit 1; }

if [ "$SKIP_ENV_UPDATE" = false ]; then
    if conda run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
        conda env update -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml"
    else
        conda env create -y -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml"
    fi
fi

if [ ! -d "$LIBERO_HOME/.git" ]; then
    if [ -e "$LIBERO_HOME" ] && [ -n "$(find "$LIBERO_HOME" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "ERROR: --libero-home exists but is not a Git checkout: $LIBERO_HOME" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$LIBERO_HOME")"
    git clone "$LIBERO_REPOSITORY" "$LIBERO_HOME"
    git -C "$LIBERO_HOME" checkout --detach "$LIBERO_REVISION"
else
    CURRENT_REVISION="$(git -C "$LIBERO_HOME" rev-parse HEAD)"
    if [ "$CURRENT_REVISION" != "$LIBERO_REVISION" ]; then
        echo "ERROR: existing LIBERO checkout is at $CURRENT_REVISION" >&2
        echo "Expected pinned revision $LIBERO_REVISION; use a fresh --libero-home path." >&2
        exit 1
    fi
fi

# Avoid robosuite's unbounded opencv-python dependency and LIBERO's obsolete
# training requirements. All runtime dependencies are pinned in environment.yml.
conda run -n "$ENV_NAME" python -m pip install --no-deps "robosuite==1.4.0"
conda run -n "$ENV_NAME" python -m pip install \
    --no-deps \
    --config-settings editable_mode=compat \
    --editable "$LIBERO_HOME"
conda run -n "$ENV_NAME" python "$PROJECT_ROOT/scripts/configure_libero.py" \
    --libero-home "$LIBERO_HOME" \
    --config-dir "$PROJECT_ROOT/.libero" \
    --env-file "$PROJECT_ROOT/.env.libero" \
    --env-name "$ENV_NAME" \
    --mujoco-gl "$MUJOCO_GL_BACKEND"

export LIBERO_HOME
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/.libero"
export MUJOCO_GL="$MUJOCO_GL_BACKEND"
export PYTHONPATH="$PROJECT_ROOT:$LIBERO_HOME${PYTHONPATH:+:$PYTHONPATH}"

SMOKE_ARGS=()
[ "$RUN_SIMULATOR_SMOKE" = true ] && SMOKE_ARGS+=(--simulator)
conda run -n "$ENV_NAME" python "$PROJECT_ROOT/scripts/smoke_test_libero.py" "${SMOKE_ARGS[@]}"

echo "Unified NeuroVLA environment is ready: $ENV_NAME"
echo "Run evaluation with:"
echo "  bash scripts/run_brain_inspired_scripts/run_eval_libero.sh --pretrained /path/to/checkpoint"
