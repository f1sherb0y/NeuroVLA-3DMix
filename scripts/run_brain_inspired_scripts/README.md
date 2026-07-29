# Brain-Inspired Scripts (NeuroVLA + R-STDP)

Minimal training & evaluation pipeline for the brain-inspired VLA stack in AlphaBrain:

1. **`run_neurovla_pretrain.sh`** — Pre-train NeuroVLA with standard backprop
2. **`run_stdp_finetune.sh`** — Hybrid R-STDP fine-tuning on the SNN action head
3. **`run_eval_libero.sh`** — Evaluate a NeuroVLA checkpoint on LIBERO (with optional online STDP)

All paths are resolved relative to the project root, so scripts can be invoked from anywhere.

---

## Quick Start

### Step 0 — Create the unified train/eval environment

```bash
bash scripts/setup_neurovla_env.sh
```

This creates/updates the `neurovla` Conda environment, checks out the pinned
LIBERO revision under `third_party/LIBERO`, installs it without downgrading the
training stack, applies the required PyTorch 2.6 init-state compatibility patch,
and writes the non-interactive simulator config to `.libero/`.
Use `--simulator-smoke` to create and step one headless simulator without
loading a policy checkpoint.

Download the official self-contained NeuroVLA checkpoint (about 8.2 GB):

```bash
bash scripts/download_neurovla_checkpoint.sh
```

The checkpoint is stored at `~/models/neurovla-libero-all4suite/` by default.
Use `--models-dir /another/model/root` to select a different parent directory.

### Step 1 — Pre-train NeuroVLA

```bash
bash scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh
bash scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh --steps 50000 --run-id my_pretrain
```

Checkpoints → `results/training/<run_id>/checkpoints/steps_XXXX/`.

### NeuroVLA + 3D-MIX

Install the pinned VGGT package through `scripts/setup_neurovla_env.sh`, then
download the frozen official encoder and launch the matching GA2 recipe:

```bash
bash scripts/download_vggt_checkpoint.sh --models-dir ~/models
export VGGT_MODEL_PATH="$HOME/models/VGGT-1B"
export CONFIG_YAML=configs/finetune_config_ga2.yaml

bash scripts/run_brain_inspired_scripts/run_neurovla_3d_mix_pretrain.sh \
    --dataset libero_all --gpus 4 --batch-size 8 --steps 50000 \
    --run-id neurovla_3d_mix_libero_all_h100x4_bs8_ga2
```

The Qwen, Q-Former, GRU/FiLM, SNN, data mixture, action loss, and schedule match
the baseline. The added frozen VGGT-1B encoder processes the same primary and
wrist views at 518 x 518, and trainable layer-wise gates append geometry tokens
before the Q-Former. See `docs/quickstart/neurovla_3d_mix.md` for the exact
formula, token pooling setting, memory implications, and evaluation command.

### Step 2 — Hybrid R-STDP Fine-tuning

```bash
bash scripts/run_brain_inspired_scripts/run_stdp_finetune.sh \
    --pretrained results/training/my_pretrain/checkpoints/steps_50000 \
    --steps 10000 \
    --run-id my_stdp_ft
```

### Step 3 — Evaluate on LIBERO

```bash
# baseline eval on libero_goal
bash scripts/run_brain_inspired_scripts/run_eval_libero.sh \
    --pretrained ~/models/neurovla-libero-all4suite

# all 4 suites, 50 trials per task
bash scripts/run_brain_inspired_scripts/run_eval_libero.sh \
    --pretrained results/training/my_stdp_ft/checkpoints/steps_10000 \
    --suite all --trials 50

# all 4 suites sharded across 8 GPUs (videos disabled by default)
python scripts/run_brain_inspired_scripts/run_eval_libero_multi_gpu.py \
    --pretrained ~/models/neurovla-libero-all4suite \
    --suite all --trials 50 --gpus 0,1,2,3,4,5,6,7

# with online STDP test-time adaptation
bash scripts/run_brain_inspired_scripts/run_eval_libero.sh \
    --pretrained results/training/my_stdp_ft/checkpoints/steps_10000 \
    --online-stdp
```

Evaluation results → `results/evaluation/brain_inspired_eval_<timestamp>/<suite>/`.

---

## CLI Reference

### `run_neurovla_pretrain.sh` / `run_stdp_finetune.sh`

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--mode <name>` | Mode in `configs/finetune_config.yaml` | `neuro_vla` / `neuro_vla_stdp` |
| `--steps <N>` | Override `trainer.max_train_steps` | from config |
| `--run-id <str>` | Override `run_id` (checkpoint folder) | from config |
| `--gpus <N>` | Number of GPUs for `accelerate launch` | 4 |
| `--pretrained <path>` | Override `trainer.pretrained_checkpoint` | *(stdp only)* from config |

### `run_eval_libero.sh`

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--pretrained <path>` | Checkpoint to evaluate (required) | — |
| `--suite <name>` | `libero_goal` / `libero_spatial` / `libero_object` / `libero_10` / `all` | `libero_goal` |
| `--task-ids <ids>` | Comma-separated suite task IDs, such as `0,1,2` | all tasks |
| `--trials <N>` | Trials per task | 10 |
| `--seed <N>` | Random seed | 7 |
| `--gpu <id>` | GPU id (sets `CUDA_VISIBLE_DEVICES`) | 0 |
| `--video-out <dir>` | Output directory | `results/evaluation/brain_inspired_eval_<timestamp>` |
| `--online-stdp` | Enable online R-STDP test-time adaptation | off |
| `--no-video` | Skip replay video encoding and frame retention | off |
| `--dry-run` | Validate paths/environment and print commands without loading the model | off |

The wrapper reads `.env.libero` generated by the setup script and defaults to
the unified `neurovla` environment. Set `LIBERO_HOME`, `LIBERO_CONFIG_PATH`,
`MUJOCO_GL`, or `CONDA_ENV` explicitly to override those defaults.

### Multi-GPU evaluation

`run_eval_libero_multi_gpu.py` assigns every task to exactly one GPU process,
streams prefixed logs from all workers, and merges their per-task JSON results.
For all four suites, at least four GPUs are required. With eight GPUs, each
suite is split into tasks `0-4` and `5-9`. Replay videos are disabled by default;
pass `--save-videos` when they are required.

Use `--resume --output <existing-output>` after an interrupted multi-GPU run.
Completed shards are reused and only missing shards are launched.

---

## What each script launches

| Script | Python entrypoint | Purpose |
|:-------|:------------------|:--------|
| `run_neurovla_pretrain.sh` | `AlphaBrain/training/train_alphabrain.py` | End-to-end backprop training (VLM + QFormer + SNN + Edit-GRU) |
| `run_stdp_finetune.sh`     | `AlphaBrain/training/train_stdp.py`      | Load pretrained ckpt → R-STDP hybrid update on SNN head |
| `run_eval_libero.sh`       | `benchmarks/LIBERO/eval/eval_libero_online_stdp.py` | In-process LIBERO eval (no WebSocket server), optional online STDP |

---

## Related Components

| Component | Path |
|:----------|:-----|
| NeuroVLA framework | `AlphaBrain/model/framework/NeuroVLA.py` |
| SNN action model | `AlphaBrain/model/modules/action_model/spike_action_model_multitimestep.py` |
| STDP optimizer / learner / online adapter | `AlphaBrain/model/modules/action_model/stdp/` |
| Config (mode router) | `configs/finetune_config.yaml` |
| NeuroVLA architecture defaults | `configs/models/neurovla.yaml` |

---

## A Note from the Team

This is the **first open attempt at brain-inspired models in embodied AI**. The exploration of brain-inspired VLA has just begun. We are releasing the current training / evaluation pipeline and preliminary results to the community so we can move this frontier forward together.

Many directions have not been fully tuned yet:

- **libero-long** and other long-horizon tasks still have significant room for improvement
- The search space for R-STDP hyper-parameters (α/β, A+/A-, τ, warmup) is far from exhausted
- The SNN architecture (depth, LIF parameters, spike encoding) can continue to evolve
- Self-supervised reward design for online STDP has many more variants worth exploring
- Cross-suite and cross-task transfer has not been systematically evaluated

If you have ideas — training tricks, architecture changes, new benchmarks, or interesting ablations — **please reach out**. Let's pave the road for brain-inspired embodied intelligence together!
