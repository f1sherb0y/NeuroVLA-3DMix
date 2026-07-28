# NeuroVLA + 3D-MIX

This variant inserts the 3D-MIX semantic-conditioned gate between Qwen2.5-VL
hidden states and NeuroVLA's layer-wise Q-Former. The GRU/FiLM cerebellar path,
SNN action head, action horizon, loss, and LIBERO input views remain unchanged.

## Architecture

For each cortical forward pass:

1. Frozen VGGT-1B extracts geometry-aware patch tokens from the original
   primary and wrist images at 518 x 518.
2. A learned projection maps VGGT's 2048-dimensional tokens into Qwen's hidden
   space.
3. For every selected Qwen layer, an independent gate implements equations
   (2)-(5) from the 3D-MIX paper:

   `gate = sigmoid(W_gate [mean(H); F_geo])`

   `F_fused = gate * W_s mean(H) + (1 - gate) * W_g F_geo`

4. The fused tokens are appended to the unmodified Qwen hidden sequence and
   consumed by the existing Q-Former.

The paper-faithful default retains each view's full 37 x 37 VGGT grid, giving
2738 geometry tokens for the two LIBERO views. If memory is constrained, set
`framework.three_d_mix.max_geometry_tokens=512` to pool each view to 16 x 16;
record that change when comparing against the baseline because it is an
additional ablation.

## Server setup

Update the unified environment after pulling the repository:

```bash
bash scripts/setup_neurovla_env.sh --skip-env-update
bash scripts/download_vggt_checkpoint.sh --models-dir ~/models
export VGGT_MODEL_PATH="$HOME/models/VGGT-1B"
```

The downloader is pinned to the official `facebook/VGGT-1B` revision and
streams progress. VGGT is frozen during training but is included in each saved
NeuroVLA3DMix checkpoint, so evaluation does not need the original VGGT weight
directory. The pinned VGGT Python package must still be installed.

The original `facebook/VGGT-1B` checkpoint is licensed CC BY-NC 4.0. Use it
only where that non-commercial license is acceptable.

## Matching comparison run

This command mirrors the baseline four-GPU, batch-8, gradient-accumulation-2,
50k-step recipe:

```bash
export CONFIG_YAML=configs/finetune_config_ga2.yaml
export VGGT_MODEL_PATH="$HOME/models/VGGT-1B"

bash scripts/run_brain_inspired_scripts/run_neurovla_3d_mix_pretrain.sh \
    --dataset libero_all \
    --gpus 4 \
    --batch-size 8 \
    --steps 50000 \
    --run-id neurovla_3d_mix_libero_all_h100x4_bs8_ga2
```

VGGT adds roughly 1.26B frozen parameters and substantial compute. Do not
assume the baseline batch size will fit on GPUs with less memory than the H100
training instance; reduce per-device batch size and increase gradient
accumulation by the same factor to preserve the global batch size.

## Evaluation

The normal single- and multi-GPU evaluators inspect `framework_config.yaml` and
construct `NeuroVLA3DMix` automatically:

```bash
python scripts/run_brain_inspired_scripts/run_eval_libero_multi_gpu.py \
    --pretrained /path/to/neurovla_3d_mix/checkpoints/steps_50000 \
    --suite all \
    --trials 50 \
    --gpus 0,1,2,3,4,5,6,7
```

Each evaluation GPU loads a complete Qwen + NeuroVLA + VGGT model.
