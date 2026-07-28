# RL-Token

Two-phase TD3 online RL fine-tuning on a pretrained QwenOFT VLA — encoder pretrain + off-policy rollouts.

---

## Prerequisites

- A pretrained VLA checkpoint (see [Baseline VLA](baselineVLA.md)).
- LIBERO in a separate conda env; `.env` has `LIBERO_PYTHON` and `LIBERO_HOME`.
- **6 GPUs** (5 rollout + 1 train) for the default. Reduce `--num_envs_per_task` for smaller setups.

---

## Train

```bash
# Phase-1: encoder pretrain (action-token bottleneck recipe)
TRACK=rlt_a bash scripts/run_rl_scripts/run_rlt_pretrain.sh 0

# Phase-2: off-policy TD3 RL over all tasks
TRACK=rlt_a bash scripts/run_rl_scripts/run_rlt_rl.sh 0
```

Checkpoints land in `results/rlt_training/rlt_a_rl_qwen_t0_<ts>/rl_offpolicy/checkpoints/`.

## Evaluate

```bash
bash scripts/run_rl_scripts/run_eval_rlt_a.sh \
    results/action_token_training_TD3/action_token_5traj_alltasks_release_0414_1727/rl_offpolicy
```

Results: `<RUN_DIR>/eval_rl_offpolicy_iter_<NNNNN>/summary.json`.

---

Full CLI reference, rollout/TD math, and release disclaimer: [`scripts/run_rl_scripts/README.md`](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/scripts/run_rl_scripts/README.md).
