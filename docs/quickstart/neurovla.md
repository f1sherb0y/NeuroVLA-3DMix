# NeuroVLA

Brain-inspired VLA: pretrain a Spiking Neural Network action head with backprop, fine-tune with hybrid R-STDP, evaluate on LIBERO.

---

## Prerequisites

```bash
# Creates one environment for both training and in-process LIBERO evaluation.
bash scripts/setup_neurovla_env.sh --simulator-smoke
conda activate neurovla
```

The setup script writes `.env.libero` with the pinned LIBERO checkout and
headless MuJoCo settings. It does not download or load a NeuroVLA checkpoint.

Download the official all-four-suite checkpoint for evaluation:

```bash
bash scripts/download_neurovla_checkpoint.sh
```

The default destination is `~/models/neurovla-libero-all4suite/`.

Hardware: 4 × A800 80 GB default. Smaller setups work with `--gpus N`.

---

## Pipeline

### 1. Pretrain

```bash
bash scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh \
    --steps 50000 --run-id my_pretrain
```

### 2. R-STDP Fine-tune

```bash
bash scripts/run_brain_inspired_scripts/run_stdp_finetune.sh \
    --pretrained results/training/my_pretrain/checkpoints/steps_50000 \
    --steps 10000 --run-id my_stdp_ft
```

### 3. Evaluate

```bash
# Validate the checkpoint, environment, and commands without loading the model
bash scripts/run_brain_inspired_scripts/run_eval_libero.sh \
    --pretrained ~/models/neurovla-libero-all4suite \
    --suite all --dry-run

# libero_goal, 10 trials/task
bash scripts/run_brain_inspired_scripts/run_eval_libero.sh \
    --pretrained results/training/my_stdp_ft/checkpoints/steps_10000

# all 4 suites, 50 trials, with online STDP test-time adaptation
bash scripts/run_brain_inspired_scripts/run_eval_libero.sh \
    --pretrained results/training/my_stdp_ft/checkpoints/steps_10000 \
    --suite all --trials 50 --online-stdp
```

Results: `results/evaluation/brain_inspired_eval_<timestamp>/<suite>/`.

The VGGT-based comparison model has a separate architecture and matching
training recipe: [NeuroVLA + 3D-MIX](neurovla_3d_mix.md).

For an all-four-suite evaluation sharded across eight GPUs:

```bash
python scripts/run_brain_inspired_scripts/run_eval_libero_multi_gpu.py \
    --pretrained ~/models/neurovla-libero-all4suite \
    --suite all \
    --trials 50 \
    --gpus 0,1,2,3,4,5,6,7
```

This evaluates each task exactly once, streams all worker logs, and writes an
aggregate `eval_results.json`. Videos are disabled by default for throughput;
add `--save-videos` to retain every rollout.

---

Full CLI reference and the team's open-research notes: [`scripts/run_brain_inspired_scripts/README.md`](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/scripts/run_brain_inspired_scripts/README.md).
