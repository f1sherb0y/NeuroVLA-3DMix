# NeuroVLA-3DMix Project Handoff

Last updated: 2026-07-30 (Asia/Shanghai)

This document records the working context for the NeuroVLA-3DMix project: the
research objective, the verified NeuroVLA architecture, environment and
checkpoint conventions, training and evaluation workflows, the 3D-MIX design,
bugs already fixed, current server state, and the next work to perform.

## 1. Research objective

The primary goal is a controlled comparison between:

1. The existing NeuroVLA baseline.
2. The same NeuroVLA model with a paper-faithful 3D-MIX bridge backed by a
   frozen VGGT-1B geometry encoder.

Both models should use the same LIBERO data mixture, action objective, training
schedule, hardware allocation, and evaluation protocol. The active experiment
trains both variants on one 8 x H100 node with per-device batch 8, gradient
accumulation 1, global batch 64, and 50,000 optimizer steps, then evaluates both
over all four LIBERO suites.

The user does not train or evaluate NeuroVLA on the local workstation. The
local machine is used for code inspection, unit tests, environment installation,
and simulator smoke tests only. Real training and full evaluation happen on a
GPU server because the local machine does not have enough VRAM.

NeuroVLA also supports hybrid R-STDP fine-tuning and online R-STDP evaluation,
but those are separate experimental variables. The current baseline-versus-
3D-MIX comparison uses the standard backprop pretraining path and evaluation
with online STDP disabled unless the user explicitly requests an STDP study.

## 2. Repository state

- Public GitHub repository: `https://github.com/f1sherb0y/NeuroVLA-3DMix`
- Branch: `main`
- Remote name: `origin`
- Local workspace: `/home/fish/Documents/ncrc/NeuroVLA-VGGT`
- Current handoff base commit: `076ceeb`
- Current server checkout used by the user: `/home/junyu/NeuroVLA-3DMix`
- Server account that owns Conda, models, and results: `junyu`

Important commits, oldest to newest:

| Commit | Purpose |
| --- | --- |
| `c4406fb` | Correct NeuroVLA config, gradient scaling, data aliases, and action chunking |
| `f28b3f7` | Add one reproducible Conda environment for training and evaluation |
| `b926d0e` | Patch LIBERO init-state loading for PyTorch 2.6 |
| `1bc764c` | Add pinned official NeuroVLA checkpoint downloader |
| `e9dee9c` | Stream checkpoint download progress |
| `b90d3b4` | Add sharded multi-GPU LIBERO evaluation |
| `9fd5e52` | Add the VGGT-backed NeuroVLA3DMix architecture and recipe |
| `84b2bee` | Simplify 3D-MIX to the direct paper path and remove defensive branches |
| `09d96ae` | Fix EGL logical-device mapping for multi-GPU evaluation |
| `9f39a09` | Document the NeuroVLA-3DMix project handoff |
| `076ceeb` | Correct Q-Former padding masks and extend them for 3D-MIX tokens |

Before starting new work, run:

```bash
git status --short
git pull --ff-only
```

Do not assume the repository is under `/home/yhcheng`. All scripts added for
this project resolve paths relative to the checkout or use `$HOME` and explicit
environment variables.

## 3. Source material used

The 3D-MIX design was derived from the paper material adjacent to this
repository:

- `../papers/3D-MIX/3D-MIX.md`
- `../papers/NeuroVLA/NeuroVLA.md`
- `../papers/NeuroVLA/NeuroVLA_3DMix_Integration.md`
- `../papers/NeuroVLA-3DMix`
- `../papers/VGGT/vggt`

The official 3D-MIX source repository was not available during implementation.
The paper's advertised `ZGC-EmbodyAI/3DMix-for-VLA` URL still returned 404 on
2026-07-30 and was absent from the organization's public repository list. The
implementation was checked against the paper equations, the local integration
proposal, and the official VGGT source checkout; exact unreleased-code parity
cannot be claimed.

Pinned external revisions:

| Component | Revision |
| --- | --- |
| LIBERO source | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| VGGT source | `a288dd0f14786c93483e45524328726ab7b1b4ce` |
| VGGT-1B weights | `860abec7937da0a4c03c41d3c269c366e82abdf9` |
| Official NeuroVLA checkpoint | `d8a5f51747d54b83bb9cf5742a8a7b3236a66c70` |

VGGT-1B weights are licensed CC BY-NC 4.0. Confirm that the non-commercial
license is acceptable for every training and distribution context.

## 4. Verified official NeuroVLA checkpoint

The official self-contained checkpoint is hosted at:

```text
AlphaBrainGroup/neurovla-libero-all4suite
```

The downloader places it at:

```text
~/models/neurovla-libero-all4suite/
```

The inspected local copy is currently at:

```text
/home/fish/Documents/ncrc/projects/NeuroVLA/models/neurovla-libero-all4suite/
```

Download command:

```bash
bash scripts/download_neurovla_checkpoint.sh
```

The pinned `model.safetensors` is 8,167,582,462 bytes. The self-contained
directory includes:

- `model.safetensors`
- `framework_config.yaml`
- `dataset_statistics.json`
- `qwen_pretrained/` with Qwen config, tokenizer, processor, and chat template

Although the saved framework config contains the historical training server's
absolute Qwen path, evaluation does not need that path. The checkpoint loader
uses the bundled `qwen_pretrained/` metadata and loads the complete weights from
`model.safetensors`.

The checkpoint was inspected with `safetensors`, not inferred only from YAML.
It contains 904 tensors under these top-level modules:

| Module | Tensor keys |
| --- | ---: |
| `qwen_vl_interface` | 825 |
| `layer_qformer` | 15 |
| `action_model` | 34 |
| `edit_model` | 30 |

Measured architecture-defining tensors include:

| Tensor | Shape |
| --- | --- |
| `layer_qformer.query_tokens` | `[8, 768]` |
| `layer_qformer.proj.weight` | `[768, 2048]` |
| `action_model.model.fc1.weight` | `[1536, 768]` |
| `action_model.model.fc2.weight` | `[1536, 1536]` |
| `action_model.model.fc3.weight` | `[7, 1536]` |
| `edit_model.robot_state_encoder.weight_ih_l0` | `[192, 8]` |
| `edit_model.robot_state_encoder.weight_hh_l0` | `[192, 64]` |

The official saved config identifies this model as architecture version 2.
Its recorded training metadata says:

| Setting | Official checkpoint metadata |
| --- | --- |
| Run ID | `0421-NeuroVLA-All4Suite-bs16-sdpa` |
| Dataset mix | `libero_all` |
| GPUs | 2 in the model card/config; 4 in the later `resume_meta.json` |
| Per-device batch | 16 |
| Gradient accumulation | 1 |
| Steps | 50,000 |
| R-STDP steps | 0 |
| Attention | SDPA |

The GPU metadata is internally inconsistent, but every source agrees on 50,000
supervised-backprop steps and no R-STDP phase. The published checkpoint is used
as the existing evaluation baseline. The active newly trained comparison uses
global batch 64 on 8 H100s for both baseline and 3D-MIX; this preserves the
colleague recipe's `4 x 8 x GA2 = 64` effective batch as `8 x 8 x GA1 = 64`.

## 5. NeuroVLA architecture

The baseline implementation is in:

- `AlphaBrain/model/framework/NeuroVLA.py`
- `AlphaBrain/model/modules/projector/qformer.py`
- `AlphaBrain/model/modules/action_model/spike_action_model_multitimestep.py`
- `configs/models/neurovla.yaml`

The model has three conceptual tiers.

### 5.1 Cortical tier

- Backbone: `Qwen2.5-VL-3B-Instruct`
- Input: primary RGB view, wrist RGB view, and language instruction
- Qwen hidden dimension: 2048
- Selected hidden-state range: layers `[36:37]`, so the current config uses one
  Qwen hidden layer
- Layerwise Q-Former input dimension: 2048
- Q-Former output dimension: 768
- Learnable query tokens: 8
- Cross-attention heads: 8 in the current constructor default
- Qwen-to-action gradient scale: 0.5

Gradient scaling preserves forward values and scales only the gradient flowing
back into the selected Qwen hidden states. It uses a detach-based identity,
which avoids accumulating tensor hooks.

### 5.2 Cerebellar tier

The Q-Former produces semantic action features with shape `[B, 8, 768]`.
The edit model conditions those features on robot-state history using a
two-layer GRU and gated FiLM-style modulation.

- Robot state dimension: 8
- Action dimension: 7
- State-only channel 8 is preserved when predicted actions are rolled forward
- Edit hidden dimension: 256

### 5.3 Spinal tier

The action head is an SNN continuous regressor:

- Input dimension: 768
- Hidden dimension: 1536
- Two MLP residual blocks with leaky integrate-and-fire neurons
- Output dimension: 7
- Continuous output is read from membrane potential
- Action horizon: 8
- Training objective: L1 action loss

SNN state is reset at the start of each forward pass.

### 5.4 Action chunking

One Q-Former query corresponds to one predicted action step. The current query
count and action horizon are both 8, so one action-head pass produces the full
chunk. The implementation also handles horizons that are not exact multiples
of the query count by using ceiling division, concatenating chunks, and slicing
to the requested horizon.

## 6. Original NeuroVLA bugs already corrected

Do not reintroduce these behaviors.

### 6.1 Config did not match the official checkpoint

The earlier YAML used a misspelled `ouptput_dim`, action hidden size 1024,
state dimension 7, and action horizon 16. The official checkpoint proves the
actual architecture is:

- `output_dim: 768`
- action hidden size 1536
- state dimension 8
- action horizon 8
- architecture version 2

Legacy checkpoint configs without `architecture_version` retain version-1
compatibility.

### 6.2 Action chunk calculation could truncate or produce the wrong length

The earlier implementation used floor division and duplicated slightly
different loops in training and inference. Training and inference now share
`_predict_action_chunks`, use ceiling division, and return exactly the requested
horizon.

### 6.3 Robot state roll-forward was hard-coded

The earlier code assumed seven action channels plus one fixed channel in
multiple places. `_roll_forward_states` now derives dimensions from the model
config and preserves state-only channels.

### 6.4 Q-Former gradient scaling was effectively broken

The old implementation looked under the wrong config path and had disabled or
hook-based behavior. It now reads `framework.layer_qformer.grad_scale` and uses
a stable value-preserving expression.

### 6.5 Q-Former output-dimension spelling

New configs use `output_dim`. The loader still accepts `ouptput_dim` for old
saved checkpoints.

### 6.6 Inference did unnecessary language-model work

`predict_action` disables cache, requests hidden states, and uses
`logits_to_keep=1`; it does not calculate an unused language-model loss.

### 6.7 LIBERO data naming and root variables

- Training modes use `LEROBOT_LIBERO_DATA_DIR`, not `LIBERO_DATA_ROOT`.
- `libero_10` is a supported alias for the long-horizon suite.
- `libero_all` includes object, goal, spatial, and `libero_10` datasets with
  equal mixture weights.

### 6.8 PyTorch 2.6 broke LIBERO init-state loading

PyTorch 2.6 changed `torch.load` to `weights_only=True` by default. Official
LIBERO init-state files include NumPy objects. The pinned patch changes the
trusted official init-state load to:

```python
torch.load(init_states_path, weights_only=False)
```

The setup script applies `patches/libero-pytorch-2.6.patch` to the pinned LIBERO
checkout.

### 6.9 Multi-GPU MuJoCo EGL mapping was wrong

The first eight-GPU run exposed one physical GPU per process with
`CUDA_VISIBLE_DEVICES=<physical ID>` but also passed that physical ID to EGL.
Inside the isolated process, the visible GPU is logical device 0. GPUs 1-7
therefore failed with messages such as:

```text
The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and 0
```

`run_eval_libero.sh` now always uses `MUJOCO_EGL_DEVICE_ID=0` after isolating a
worker's physical GPU.

### 6.10 Q-Former conditioning masks were invalid and unused

The Q-Former previously expanded a `[B, L]` mask to `[B, 1, L]`, which
`nn.MultiheadAttention` rejects for `key_padding_mask`, and passed the keep-mask
with the opposite boolean polarity. Baseline NeuroVLA also did not forward the
Qwen attention mask to the Q-Former. Commit `076ceeb` now:

- passes a two-dimensional key-padding mask;
- inverts Qwen's `1 = visible` convention to PyTorch's `True = ignore`;
- forwards the original semantic mask in baseline NeuroVLA; and
- appends visible mask entries for every 3D-MIX geometry token.

Single-sample inference without padding is unchanged. Batched training now
excludes left-padding states while retaining every appended geometry token.

## 7. NeuroVLA3DMix architecture

The new variant is implemented in:

- `AlphaBrain/model/framework/NeuroVLA3DMix.py`
- `AlphaBrain/model/modules/projector/three_d_mix.py`
- `configs/models/neurovla_3d_mix.yaml`

The design changes only the bridge between Qwen hidden states and the existing
Q-Former. The GRU, FiLM path, SNN action head, action horizon, state handling,
and L1 objective are unchanged.

### 7.1 Geometry extraction

- Encoder: official VGGT-1B aggregator
- VGGT parameters: frozen and kept in evaluation mode
- Inputs: original-resolution primary and wrist RGB images
- Preprocessing: preserve aspect ratio, resize dimensions to multiples of the
  14-pixel patch size, and center-pad to 518 x 518 with white
- Feature source: final cached VGGT aggregator output
- Camera/register tokens are removed using `patch_start_idx`
- Each 518 x 518 view yields a 37 x 37 patch grid
- Two LIBERO views yield 2,738 geometry tokens
- VGGT output dimension after concatenated frame/global features: 2048

The data loader retains a separate `vggt_image` list when
`include_vggt_images: true`. This prevents VGGT from receiving the 224 x 224
images prepared for Qwen.

### 7.2 Paper equations

VGGT features are projected once into the Qwen hidden space:

```text
F_geo = W_proj F_VGGT
```

For every selected Qwen layer, the implementation computes a masked semantic
mean and an independent per-token, per-channel gate:

```text
s_global = masked_mean(H)
g = sigmoid(W_gate [s_global; F_geo])
F_fused = g * W_s(s_global) + (1 - g) * W_g(F_geo)
H_cond = [H; F_fused]
M_cond = [M_qwen; 1_N]
```

`H_cond` replaces `H` only as input to the existing Layerwise Q-Former. The
matching conditioning mask excludes Qwen padding and marks all `N` appended
geometry tokens as visible.

The code supports one fusion layer per selected Qwen layer. The current
NeuroVLA config selects only layer 36, so the current experiment has one gated
fusion layer. If NeuroVLA later selects multiple hidden layers, each receives
its own gate while sharing the one extracted/projected VGGT feature tensor.

### 7.3 Gradient behavior

The baseline Qwen interface gradient scale of 0.5 is applied before 3D-MIX.
This scales gradients returning to Qwen but does not attenuate gradients for
the trainable geometry projection or gated-fusion parameters.

### 7.4 Deliberate simplicity

The final implementation follows the direct paper path. It does not contain an
optional geometry-token pooling mode or extensive defensive input branches.
All 2,738 geometry tokens are retained.

### 7.5 Checkpoint behavior

VGGT is frozen but remains a registered submodule, so trained NeuroVLA3DMix
checkpoints include its aggregator weights. A saved 3D-MIX checkpoint is
self-contained for weights:

- Training from scratch needs the external VGGT-1B checkpoint.
- Evaluation of a trained 3D-MIX checkpoint does not need the external VGGT
  weight directory.
- Evaluation still needs the pinned VGGT Python package to construct the
  aggregator architecture.

`BaseFramework.from_pretrained` sets
`initialize_vggt_from_checkpoint: true` while loading a NeuroVLA3DMix directory,
constructs the aggregator skeleton, and fills it from the saved state dict.

## 8. Reproducible environment

The single Conda environment is named `neurovla` and is defined by
`environment.yml`.

Core versions:

| Package/tool | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| CUDA nvcc | 12.4 |
| transformers | 4.57.0 |
| accelerate | 1.5.2 |
| DeepSpeed | 0.16.9 |
| NumPy | 1.26.4 |
| MuJoCo | 2.3.7 |
| robosuite | 1.4.0 |
| opencv-python-headless | 4.11.0.86 |

Fresh setup:

```bash
bash scripts/setup_neurovla_env.sh --simulator-smoke
conda activate neurovla
```

Existing environment after pulling code:

```bash
bash scripts/setup_neurovla_env.sh --skip-env-update --simulator-smoke
```

The setup script:

1. Creates or updates `neurovla` from `environment.yml`.
2. Clones the pinned LIBERO revision into `third_party/LIBERO` by default.
3. Applies the PyTorch 2.6 compatibility patch.
4. Installs robosuite and LIBERO without their obsolete dependency sets.
5. Installs the pinned VGGT source without replacing headless OpenCV.
6. Writes `.libero/config.yaml`.
7. Writes `.env.libero` with absolute paths and the concrete interpreter.
8. Runs import checks and, with `--simulator-smoke`, creates, renders, and steps
   one headless LIBERO simulator.

Successful smoke tests were performed both locally and on the server. Expected
non-fatal warnings include:

- robosuite's missing private macro file warning
- Gym's unmaintained-project warning

Do not replace Gym with Gymnasium casually; LIBERO/robosuite compatibility is
the priority for this pinned environment.

Local validation does not include loading VGGT-1B weights, loading a complete
NeuroVLA3DMix model, or running full policy evaluation.

## 9. Model and dataset locations

Confirmed training-server layout under `/home/junyu`:

```text
$HOME/
  NeuroVLA-3DMix/
  models/
    Qwen2.5-VL-3B-Instruct/
    VGGT-1B/
    neurovla-libero-all4suite/
  datasets/libero/
    libero_goal_no_noops_1.0.0_lerobot/
    libero_10_no_noops_1.0.0_lerobot/
    libero_object_no_noops_1.0.0_lerobot/
    libero_spatial_no_noops_1.0.0_lerobot/
```

Required environment variables for the active training run:

```bash
export CONDA_ENV=neurovla
export NO_ALBUMENTATIONS_UPDATE=1
export PRETRAINED_MODELS_DIR="$HOME/models"
export VGGT_MODEL_PATH="$HOME/models/VGGT-1B"
export LEROBOT_LIBERO_DATA_DIR="$HOME/datasets/libero"
```

The launch wrapper already defaults to `configs/finetune_config.yaml`,
`configs/deepspeed/accelerate_zero2.yaml`, and process port 29500. No NCCL
InfiniBand overrides or explicit `MAIN_PROCESS_PORT` are required for the
single-node 8-H100 job.

`PRETRAINED_MODELS_DIR` must contain `Qwen2.5-VL-3B-Instruct/`. The config
builder joins those two components when resolving the baseline Qwen model.

Download VGGT with visible progress:

```bash
bash scripts/download_vggt_checkpoint.sh --models-dir "$HOME/models"
```

Download the official baseline checkpoint with visible progress:

```bash
bash scripts/download_neurovla_checkpoint.sh --models-dir "$HOME/models"
```

Training data and native LIBERO simulator assets are different things:

- `LEROBOT_LIBERO_DATA_DIR` points to LeRobot-format training datasets.
- Every downloaded training subset must contain `meta/modality.json`; the
  canonical file is `benchmarks/LIBERO/train/modality.json`.
- `LIBERO_HOME` points to the pinned simulator source and its BDDL/init/assets.
- `.env.libero` is generated for evaluation and should not be committed.

## 10. Baseline training workflow

The colleague's historical server command used `/home/yhcheng`, Conda
environment `alphabrain`, four H100 GPUs, per-device batch 8, gradient
accumulation 2, and 50,000 steps. The exact historical command was:

```bash
HOME=/home/yhcheng
source /home/yhcheng/miniconda3/etc/profile.d/conda.sh
cd /home/yhcheng/proj/AlphaBrain
CONFIG_YAML=configs/finetune_config_ga2.yaml
conda activate alphabrain
bash scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh \
    --dataset libero_all \
    --gpus 4 \
    --batch-size 8 \
    --steps 50000 \
    --run-id neurovla_pretrain_libero_all_h100x4_bs8_ga2
```

Do not copy the hard-coded home paths or old environment name. On the active
8-H100 server, preserve the historical global batch of 64 by replacing
`4 x batch 8 x GA2` with `8 x batch 8 x GA1`. The active baseline launch is:

```bash
export CONDA_ENV=neurovla
export NO_ALBUMENTATIONS_UPDATE=1
export PRETRAINED_MODELS_DIR="$HOME/models"
export VGGT_MODEL_PATH="$HOME/models/VGGT-1B"
export LEROBOT_LIBERO_DATA_DIR="$HOME/datasets/libero"

bash scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh \
    --dataset libero_all \
    --gpus 8 \
    --batch-size 8 \
    --steps 50000 \
    --run-id neurovla_baseline_libero_all_h100x8_bs8_ga1
```

The active recipe resolves to:

| Setting | Value |
| --- | ---: |
| GPUs | 8 |
| Per-device batch | 8 |
| Gradient accumulation | 1 |
| Effective global batch | 64 |
| Steps | 50,000 |
| Visible training samples | 3.2 million |
| Save interval | 10,000 |
| Attention implementation | SDPA |
| Base LR | `2.5e-5` |
| Qwen LR | `1.0e-5` |
| Q-Former LR | `5.0e-5` |
| Action model LR | `1.0e-4` |

Keep SDPA for the controlled comparison. The project notes report unstable
training changes when substituting Flash Attention 2 without retuning.

Checkpoints are written under:

```text
results/training/<run-id>/checkpoints/steps_<N>/
```

## 11. NeuroVLA3DMix training workflow

Use the same 8-H100 schedule and change only the model launcher:

```bash
bash scripts/run_brain_inspired_scripts/run_neurovla_3d_mix_pretrain.sh \
    --dataset libero_all \
    --gpus 8 \
    --batch-size 8 \
    --steps 50000 \
    --run-id neurovla_3d_mix_libero_all_h100x8_bs8_ga1
```

The 3D-MIX mode adds only:

- `framework.name: NeuroVLA3DMix`
- `include_vggt_images: true`
- `three_d_mix` optimizer group at LR `1.0e-4`
- frozen VGGT feature extraction and gated fusion

VGGT adds roughly 1.26B frozen parameters and substantial compute. The
8-H100 batch-8 run has not yet completed a real model step. If it runs out of
memory, switch both experiments to `configs/finetune_config_ga2.yaml` with
per-device batch 4, preserving `8 x 4 x GA2 = 64`. Record any recipe change.

### 11.1 Training checkpoint and resume behavior

The training wrappers are not automatically resumable. Current defaults are:

```text
trainer.is_resume = false
trainer.save_training_state = false
```

Re-running the same command and run ID therefore starts from step 0. Each
10,000-step checkpoint contains model weights and `resume_meta.json`, but no
optimizer, scheduler, or RNG state. The trainer contains opt-in warm/full resume
code, but the shell wrappers expose no `--resume` flag and the full-state path
has not been validated for these models. Do not rely on automatic recovery from
a preempted cloud task without implementing and testing it first.

### 11.2 First 8-H100 training launch incident

On 2026-07-30 the baseline launch reached dataset construction on all eight
ranks and failed before step 0 because the downloaded LeRobot subsets lacked
AlphaBrain's required `meta/modality.json`. The first reported path was:

```text
$HOME/datasets/libero/libero_object_no_noops_1.0.0_lerobot/meta/modality.json
```

This was a data-preparation issue, not a model, DeepSpeed, NCCL, or GPU failure.
Repair all four subsets once with:

```bash
cd "$HOME/NeuroVLA-3DMix"
for suite in spatial object goal 10; do
    install -D -m 0644 \
        benchmarks/LIBERO/train/modality.json \
        "$HOME/datasets/libero/libero_${suite}_no_noops_1.0.0_lerobot/meta/modality.json"
done
```

Albumentations version-check warnings caused by the server's blocked outbound
network are non-fatal. Set `NO_ALBUMENTATIONS_UPDATE=1` to suppress them. No
checkpoint or training progress was produced by this failed launch, so the same
run ID can be reused after repairing the metadata.

## 12. Evaluation workflow

Evaluation is in-process: one Python process owns both the policy and its
LIBERO simulator. There is no policy WebSocket server in this workflow.

The default evaluator passes `--no-stdp`. Add `--online-stdp` only for a
separate test-time adaptation experiment. Do not mix online STDP into one side
of the baseline-versus-3D-MIX comparison.

### 12.1 Single GPU

```bash
bash scripts/run_brain_inspired_scripts/run_eval_libero.sh \
    --pretrained "$HOME/models/neurovla-libero-all4suite" \
    --suite libero_goal \
    --trials 50 \
    --gpu 0 \
    --no-video
```

The wrapper loads `.env.libero`, exposes the selected physical GPU through
`CUDA_VISIBLE_DEVICES`, and assigns MuJoCo EGL to logical device 0.

### 12.2 Eight GPUs over all four suites

```bash
python scripts/run_brain_inspired_scripts/run_eval_libero_multi_gpu.py \
    --pretrained "$HOME/models/neurovla-libero-all4suite" \
    --suite all \
    --trials 50 \
    --gpus 0,1,2,3,4,5,6,7 \
    --output "$HOME/NeuroVLA-3DMix/results/evaluation/neurovla_all_8gpu"
```

The eight-worker plan is:

| GPU | Suite | Tasks |
| ---: | --- | --- |
| 0 | `libero_goal` | 0-4 |
| 1 | `libero_goal` | 5-9 |
| 2 | `libero_spatial` | 0-4 |
| 3 | `libero_spatial` | 5-9 |
| 4 | `libero_object` | 0-4 |
| 5 | `libero_object` | 5-9 |
| 6 | `libero_10` | 0-4 |
| 7 | `libero_10` | 5-9 |

Each shard runs 5 tasks x 50 trials = 250 episodes. The complete evaluation is
40 tasks x 50 trials = 2,000 episodes. Videos are disabled by default in the
multi-GPU launcher.

Output layout:

```text
<output>/
  logs/<shard>.log
  shards/<shard>/<suite>/eval_results.json
  <suite>/eval_results.json
  eval_results.json
```

The per-suite and root aggregate files are written only after all shards
succeed.

### 12.3 Root-launched server tasks

The task runner starts as root, but root cannot reliably see the user's Conda
installation and should not own result files. Run the complete command as
`junyu`:

```bash
runuser -u junyu -- env HOME=/home/junyu /bin/bash -lc '
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate neurovla
cd "$HOME/NeuroVLA-3DMix"
git pull --ff-only

exec python scripts/run_brain_inspired_scripts/run_eval_libero_multi_gpu.py \
    --pretrained "$HOME/models/neurovla-libero-all4suite" \
    --suite all \
    --trials 50 \
    --gpus 0,1,2,3,4,5,6,7 \
    --output "$HOME/NeuroVLA-3DMix/results/evaluation/neurovla_all_8gpu_retry"
'
```

Use a new output directory for a clean retry. Do not delete old output from a
root task unless the user explicitly requests it.

### 12.4 Resume semantics

`--resume` reuses a shard only when this exact file exists:

```text
shards/<shard>/<suite>/eval_results.json
```

Empty directories are not completed shards. Check for reusable results with:

```bash
find "$HOME/NeuroVLA-3DMix/results/evaluation/neurovla_all_8gpu" \
    -type f -name eval_results.json -size +0c -print
```

If the command prints nothing, use a new output directory and run without
`--resume`. If it prints valid shard result files, use the same GPU list,
suite, trial count, checkpoint, and output directory with `--resume`.

## 13. Current server evaluation incident

On 2026-07-29 the user launched the official baseline checkpoint over eight
GPUs. The captured log is available locally at:

```text
~/Downloads/logs-acp-20260729T155501.txt.gz
```

Observed behavior:

- GPU 0 started and the log reported 250 completed `libero_goal` episodes.
- The log reported a 90.0% success rate for tasks 0-4 combined.
- GPUs 1-7 failed immediately because their physical GPU IDs were passed as EGL
  device IDs after CUDA isolation.
- The launcher refused to aggregate because seven shards failed.
- The code-side EGL bug was fixed in commit `09d96ae`.
- The user later inspected the server and reported that the created output
  folders were empty.

The server filesystem is authoritative. Because no persisted non-empty
`eval_results.json` files were observed, the recommended next evaluation is a
fresh run in `neurovla_all_8gpu_retry`, without `--resume`.

The earlier log's 90.0% number is useful diagnostic evidence that the baseline
model, checkpoint, simulator, and GPU 0 path worked. It is not a final result
unless the corresponding JSON file actually persists.

## 14. Evaluation performance expectations

During the failed run, the surviving GPU 0 shard processed 250 episodes in
about 1 hour 24 minutes. Individual successful episodes commonly took about
14-26 seconds; failed rollouts could take around 45 seconds because they run to
the horizon.

With eight healthy workers, total wall-clock time should be governed by the
slowest 250-episode shard rather than by all 2,000 episodes serially. Budget
roughly 1.5-2 hours, but task-dependent failure rates can increase this.

Increasing the number of GPUs helps by sharding independent tasks. It does not
make one rollout faster. Each worker loads a complete model and one simulator.

## 15. Evaluation of a trained 3D-MIX checkpoint

The same evaluator is used for both model types. It reads
`framework_config.yaml` and constructs `NeuroVLA3DMix` automatically when that
is the saved framework name.

```bash
python scripts/run_brain_inspired_scripts/run_eval_libero_multi_gpu.py \
    --pretrained results/training/neurovla_3d_mix_libero_all_h100x8_bs8_ga1/checkpoints/steps_50000 \
    --suite all \
    --trials 50 \
    --gpus 0,1,2,3,4,5,6,7 \
    --output results/evaluation/neurovla_3d_mix_all_8gpu
```

Each evaluation worker loads a complete Qwen + NeuroVLA + VGGT model. Plan GPU
memory accordingly.

## 16. Validation already completed

The following checks have passed locally:

- Conda environment creation/update
- AlphaBrain import
- PyTorch 2.6.0+cu124 import
- pinned VGGT source import
- pinned LIBERO configuration
- real headless LIBERO simulator create/reset/render/step smoke test
- shell syntax checks
- Python compilation for changed model modules
- config resolution for baseline and 3D-MIX in GA1 and GA2 modes
- official NeuroVLA downloader dry run and streamed-output behavior
- official VGGT downloader dry run and streamed-output behavior
- official checkpoint tensor inspection
- Q-Former gradient scaling numerical test
- Q-Former padding-mask shape, polarity, and masked-value invariance test
- 3D-MIX conditioning-mask extension shape test
- 3D-MIX gate equation numerical test
- VGGT aspect-pad preprocessing test
- self-contained VGGT checkpoint skeleton test
- baseline NeuroVLA regression tests
- eight-GPU shard-plan and result aggregation tests
- EGL logical-device mapping test

At handoff time the complete test suite has 23 passing tests:

```bash
set -a
source .env.libero
set +a
export NO_ALBUMENTATIONS_UPDATE=1
conda run --no-capture-output -n neurovla \
    python -m unittest discover -s tests -v
```

What has not been tested locally:

- full official baseline evaluation
- real NeuroVLA training
- loading VGGT-1B weights into the new architecture
- a complete NeuroVLA3DMix forward pass with real Qwen and VGGT weights
- NeuroVLA3DMix training
- NeuroVLA3DMix evaluation

## 17. Working principles and constraints

- Keep the baseline and 3D-MIX comparison controlled. Do not change unrelated
  action, state, loss, data, or schedule behavior in only one model.
- Keep the model implementation direct. The user explicitly requested clean
  code without extensive defensive branches.
- Do not add geometry to the GRU/FiLM/SNN fast path. 3D-MIX belongs between
  Qwen and the Q-Former.
- Do not resize VGGT inputs to Qwen's 224 x 224 input.
- Keep VGGT frozen unless a new experiment explicitly changes that variable.
- Do not silently pool geometry tokens; the implemented experiment uses all
  2,738 tokens.
- Keep Qwen gradient scaling separate from 3D-MIX parameter gradients.
- Preserve Qwen padding masks in baseline Q-Former conditioning and append
  visible mask entries for every 3D-MIX token.
- Preserve self-contained checkpoint loading.
- Do not hard-code another user's home directory in project scripts.
- Use `runuser` when a root-owned task launcher must execute work in the user's
  Conda environment.
- Do not claim a server result from logs alone when output JSON files did not
  persist.

## 18. Immediate next actions

1. Install and verify `meta/modality.json` in all four LeRobot LIBERO subset
   directories on the server.
2. Relaunch `neurovla_baseline_libero_all_h100x8_bs8_ga1` as `junyu` and confirm
   dataset construction, all eight ranks, the first optimizer step, and GPU
   memory use.
3. Let the baseline reach step 50,000; the sequential cloud command then starts
   `neurovla_3d_mix_libero_all_h100x8_bs8_ga1`.
4. Confirm the 3D-MIX run completes its first step at batch 8. If it OOMs, stop
   and relaunch both controlled runs with batch 4 and GA2 rather than changing
   only one side.
5. Evaluate both resulting step-50,000 checkpoints with the same 50-trial,
   four-suite protocol and online STDP disabled.
6. Rerun the official baseline checkpoint evaluation with the fixed EGL mapping
   when needed for the published-checkpoint reference result.
7. Compare overall, per-suite, and per-task success rates. Record hardware,
   batch, gradient accumulation, seed, checkpoint, code commit, and wall time.

Do not change the 3D-MIX architecture again until the user provides further
experimental instructions or a real training/evaluation failure supplies new
evidence.
