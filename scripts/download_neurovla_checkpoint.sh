#!/usr/bin/env bash
# Download the official self-contained NeuroVLA checkpoint from Hugging Face.
set -euo pipefail

REPO_ID="AlphaBrainGroup/neurovla-libero-all4suite"
DEFAULT_REVISION="d8a5f51747d54b83bb9cf5742a8a7b3236a66c70"
EXPECTED_MODEL_SIZE=8167582462

ENV_NAME="${CONDA_ENV:-neurovla}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
REVISION="$DEFAULT_REVISION"
MAX_WORKERS=8
FORCE_DOWNLOAD=false
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: bash scripts/download_neurovla_checkpoint.sh [options]

Downloads the official checkpoint to:
  ~/models/neurovla-libero-all4suite

Options:
  --models-dir DIR      Parent model directory (default: ~/models)
  --env-name NAME       Conda environment containing the hf CLI (default: neurovla)
  --revision REV        Hugging Face revision (default: pinned official revision)
  --max-workers N       Concurrent download workers (default: 8)
  --force-download      Redownload files even when already cached locally
  --dry-run             Print the resolved download without writing files
  -h, --help            Show this help

The download is public and does not require HF_TOKEN. Interrupted downloads can
be resumed by running the same command again.
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
TARGET_DIR="$MODELS_DIR/neurovla-libero-all4suite"

echo "Official checkpoint: $REPO_ID"
echo "Revision:            $REVISION"
echo "Destination:         $TARGET_DIR"

if [ "$DRY_RUN" = true ]; then
    echo "Dry run: no files were downloaded"
    exit 0
fi

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

if [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] && command -v hf >/dev/null 2>&1; then
    HF_CMD=(hf)
elif command -v conda >/dev/null 2>&1 \
    && conda run -n "$ENV_NAME" hf --help >/dev/null 2>&1; then
    # Stream stdout/stderr directly so the multi-gigabyte download progress is
    # visible instead of being buffered by `conda run` until completion.
    HF_CMD=(conda run --no-capture-output -n "$ENV_NAME" hf)
else
    echo "ERROR: cannot find the hf CLI in Conda environment: $ENV_NAME" >&2
    echo "Run: bash scripts/setup_neurovla_env.sh" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"
DOWNLOAD_ARGS=(
    download "$REPO_ID"
    --revision "$REVISION"
    --local-dir "$TARGET_DIR"
    --max-workers "$MAX_WORKERS"
)
[ "$FORCE_DOWNLOAD" = true ] && DOWNLOAD_ARGS+=(--force-download)
unset HF_HUB_DISABLE_PROGRESS_BARS
"${HF_CMD[@]}" "${DOWNLOAD_ARGS[@]}"

REQUIRED_FILES=(
    framework_config.yaml
    dataset_statistics.json
    model.safetensors
    qwen_pretrained/config.json
    qwen_pretrained/tokenizer.json
)
for REQUIRED_FILE in "${REQUIRED_FILES[@]}"; do
    [ -f "$TARGET_DIR/$REQUIRED_FILE" ] || {
        echo "ERROR: downloaded checkpoint is missing $REQUIRED_FILE" >&2
        exit 1
    }
done

if [ "$REVISION" = "$DEFAULT_REVISION" ]; then
    ACTUAL_MODEL_SIZE="$(stat -c '%s' "$TARGET_DIR/model.safetensors")"
    [ "$ACTUAL_MODEL_SIZE" = "$EXPECTED_MODEL_SIZE" ] || {
        echo "ERROR: model.safetensors has size $ACTUAL_MODEL_SIZE; expected $EXPECTED_MODEL_SIZE" >&2
        echo "Run the script again to resume the download." >&2
        exit 1
    }
fi

echo "NeuroVLA checkpoint ready: $TARGET_DIR"
echo "Evaluate with:"
echo "  bash scripts/run_brain_inspired_scripts/run_eval_libero.sh --pretrained '$TARGET_DIR'"
