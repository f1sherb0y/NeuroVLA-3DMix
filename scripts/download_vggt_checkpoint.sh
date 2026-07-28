#!/usr/bin/env bash
# Download the frozen VGGT-1B encoder used by NeuroVLA3DMix.
set -euo pipefail

REPO_ID="facebook/VGGT-1B"
DEFAULT_REVISION="860abec7937da0a4c03c41d3c269c366e82abdf9"
ENV_NAME="${CONDA_ENV:-neurovla}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
REVISION="$DEFAULT_REVISION"
MAX_WORKERS=8
FORCE_DOWNLOAD=false
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: bash scripts/download_vggt_checkpoint.sh [options]

Downloads the frozen 3D encoder to:
  ~/models/VGGT-1B

Options:
  --models-dir DIR      Parent model directory (default: ~/models)
  --env-name NAME       Conda environment containing the hf CLI (default: neurovla)
  --revision REV        Hugging Face revision (default: pinned official revision)
  --max-workers N       Concurrent download workers (default: 8)
  --force-download      Redownload files even when already cached locally
  --dry-run             Print the resolved download without writing files
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --revision) REVISION="$2"; shift 2 ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --force-download) FORCE_DOWNLOAD=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

MODELS_DIR="$(realpath -m "$MODELS_DIR")"
TARGET_DIR="$MODELS_DIR/VGGT-1B"

echo "VGGT checkpoint: $REPO_ID"
echo "Revision:        $REVISION"
echo "Destination:     $TARGET_DIR"

if [ "$DRY_RUN" = true ]; then
    echo "Dry run: no files were downloaded"
    exit 0
fi

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

if [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] && command -v hf >/dev/null 2>&1; then
    HF_CMD=(hf)
elif command -v conda >/dev/null 2>&1 \
    && conda run -n "$ENV_NAME" hf --help >/dev/null 2>&1; then
    HF_CMD=(conda run --no-capture-output -n "$ENV_NAME" hf)
else
    echo "ERROR: cannot find the hf CLI in Conda environment: $ENV_NAME" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"
DOWNLOAD_ARGS=(
    download "$REPO_ID"
    --revision "$REVISION"
    --local-dir "$TARGET_DIR"
    --max-workers "$MAX_WORKERS"
    --include config.json model.safetensors
)
[ "$FORCE_DOWNLOAD" = true ] && DOWNLOAD_ARGS+=(--force-download)
unset HF_HUB_DISABLE_PROGRESS_BARS
"${HF_CMD[@]}" "${DOWNLOAD_ARGS[@]}"

[ -f "$TARGET_DIR/config.json" ] && [ -f "$TARGET_DIR/model.safetensors" ] || {
    echo "ERROR: downloaded checkpoint is missing config.json or model.safetensors" >&2
    exit 1
}

echo "VGGT checkpoint ready: $TARGET_DIR"
echo "Set this before training from the local copy:"
echo "  export VGGT_MODEL_PATH='$TARGET_DIR'"
