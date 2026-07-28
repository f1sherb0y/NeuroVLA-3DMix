#!/bin/bash
# Offline eval for the RLT track. Re-runs the trained encoder + actor
# through `eval_libero_rlt.py` (rolls 50 episodes/task by default, returns
# per-task + overall SR in summary.json).
#
# In-training eval_sr (logged to wandb) goes through eval_helpers and is
# fine for live monitoring, but the released RL ckpts deserve a proper
# fixed-protocol eval; this script provides it.
#
# Usage:
#   # All iter ckpts of one RL run, parallel across the given GPUs
#   RUN_DIR=results/rlt_training/<run>/rl_offpolicy \
#   VLA_CKPT=results/training/Pi05-goal-5traj-openpi/checkpoints/steps_30000 \
#   GPUS="0 1 2" TASK_IDS=0 N_EPS=50 \
#       bash scripts/run_rl_scripts/run_eval_rlt.sh
#
#   # A single iter only (set ITER to the 5-digit suffix)
#   ITER=00300 RUN_DIR=... VLA_CKPT=... GPUS=0 bash scripts/run_rl_scripts/run_eval_rlt.sh
#
# Env overrides:
#   RUN_DIR        rl_offpolicy run directory containing checkpoints/
#   VLA_CKPT       frozen VLA used during RL training
#   GPUS           space-separated GPU ids, ckpts round-robin across them
#   ITER           5-digit iter suffix; if set, only that ckpt; else all iters
#   N_EPS          episodes per task (default 50)
#   TASK_IDS       comma-separated task ids (default "0")
#   SUITE          libero_goal | ... (default libero_goal)
#   NUM_WORKERS    parallel LIBERO env workers per shard (default 4)
#   BOTTLENECK_DIM / ENCODER_LAYERS / ENCODER_HEADS / ACTOR_HIDDEN_DIM /
#   REF_DROPOUT / FIXED_STD  — RLT encoder/actor architecture (must match
#                              the values used at training time)
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ── Required ─────────────────────────────────────────────────
RUN_DIR="${RUN_DIR:-}"
VLA_CKPT="${VLA_CKPT:-}"
[ -z "${RUN_DIR}" ] && { echo "ERROR: RUN_DIR is required" >&2; exit 1; }
[ -z "${VLA_CKPT}" ] && { echo "ERROR: VLA_CKPT is required" >&2; exit 1; }
[ ! -d "${VLA_CKPT}" ] && { echo "ERROR: VLA_CKPT not found: ${VLA_CKPT}" >&2; exit 1; }
[ ! -d "${RUN_DIR}/checkpoints" ] && { echo "ERROR: no checkpoints/ under ${RUN_DIR}" >&2; exit 1; }

read -r -a GPUS <<< "${GPUS:-0}"
ITER="${ITER:-}"
N_EPS="${N_EPS:-50}"
TASK_IDS="${TASK_IDS:-0}"
SUITE="${SUITE:-libero_goal}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Arch (must match run_rlt_rl.sh)
BOTTLENECK_DIM="${BOTTLENECK_DIM:-2048}"
ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
ENCODER_HEADS="${ENCODER_HEADS:-8}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-512}"
REF_DROPOUT="${REF_DROPOUT:-0.5}"
FIXED_STD="${FIXED_STD:-0.1}"

# ── Select ckpts ────────────────────────────────────────────
if [ -n "${ITER}" ]; then
    CKPTS=("rl_offpolicy_iter_${ITER}")
else
    mapfile -t CKPTS < <(ls -1 "${RUN_DIR}/checkpoints" | grep -E '^rl_offpolicy_iter_[0-9]+$' | sort)
fi
[ "${#CKPTS[@]}" -eq 0 ] && { echo "ERROR: no matching iter ckpts under ${RUN_DIR}/checkpoints" >&2; exit 1; }

AGG_DIR="${RUN_DIR}/eval_all_iters_rlt"
mkdir -p "${AGG_DIR}"

echo "============================================================"
echo " RLT offline eval"
echo "   run_dir:  ${RUN_DIR}"
echo "   ckpts:    ${#CKPTS[@]}  (${CKPTS[*]})"
echo "   GPUs:     ${GPUS[*]}"
echo "   suite:    ${SUITE}   task_ids: ${TASK_IDS}   n_eps: ${N_EPS}"
echo "   agg_dir:  ${AGG_DIR}"
echo "============================================================"

SHARD_PIDS=()
TAIL_PIDS=()
_cleanup () {
    trap - EXIT INT TERM
    for pid in "${SHARD_PIDS[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    [ "${#TAIL_PIDS[@]}" -gt 0 ] && kill "${TAIL_PIDS[@]}" 2>/dev/null || true
    pkill -TERM -u "$(id -u)" -f "AlphaBrain/training/reinforcement_learning/envs/libero_env_worker" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

launch_one () {
    local iter_name=$1
    local gpu=$2
    local ckpt="${RUN_DIR}/checkpoints/${iter_name}"
    local iter_tag="${iter_name#rl_offpolicy_}"
    local out_dir="${RUN_DIR}/eval_${iter_tag}_rlt"
    local log="${AGG_DIR}/${iter_tag}.log"

    mkdir -p "${out_dir}"
    rm -f "${out_dir}/summary.json"

    CUDA_VISIBLE_DEVICES=${gpu} setsid python AlphaBrain/training/reinforcement_learning/eval/eval_libero_rlt.py \
        --vla_ckpt "${VLA_CKPT}" \
        --action_token_ckpt "${ckpt}" \
        --suite "${SUITE}" \
        --n_eps_per_task ${N_EPS} \
        --gpu 0 \
        --task_ids "${TASK_IDS}" \
        --results_json "${out_dir}/summary.json" \
        --num_workers ${NUM_WORKERS} \
        --bottleneck_dim ${BOTTLENECK_DIM} \
        --encoder_layers ${ENCODER_LAYERS} \
        --encoder_heads ${ENCODER_HEADS} \
        --actor_hidden_dim ${ACTOR_HIDDEN_DIM} \
        --ref_dropout ${REF_DROPOUT} \
        --fixed_std ${FIXED_STD} \
        > "${log}" 2>&1 &
    local pid=$!
    SHARD_PIDS+=(${pid})
    echo "[launch] ${iter_tag} → GPU ${gpu}  pid=${pid}  log=${log}"
    (tail -F -q -n +1 "${log}" 2>/dev/null | sed -u "s/^/[${iter_tag}] /") &
    TAIL_PIDS+=($!)
}

i=0
for ckpt_name in "${CKPTS[@]}"; do
    gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
    launch_one "${ckpt_name}" "${gpu}"
    i=$(( i + 1 ))
    sleep 2
done

echo "[all launched] waiting for ${#SHARD_PIDS[@]} shards..."

fail=0
for pid in "${SHARD_PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "ERROR: shard pid ${pid} FAILED — check ${AGG_DIR}/*.log" >&2
        fail=1
    fi
done
sleep 1
kill "${TAIL_PIDS[@]}" 2>/dev/null || true

echo "============================================================"
echo " Aggregating per-iter SR:"
python - <<PY
import json, os, re, glob
run_dir = "${RUN_DIR}"
agg_dir = "${AGG_DIR}"
rows = []
for d in sorted(glob.glob(os.path.join(run_dir, "eval_iter_*_rlt"))):
    m = re.search(r"eval_(iter_\d+)_rlt", d)
    if not m: continue
    it = m.group(1)
    summary = os.path.join(d, "summary.json")
    if not os.path.isfile(summary):
        rows.append((it, None, "NO SUMMARY"))
        continue
    with open(summary) as f:
        data = json.load(f)
    entry = data[-1] if isinstance(data, list) else data
    rows.append((it, entry.get("overall_sr"), entry.get("per_task_sr")))
print(f"{'iter':>12}  {'overall_sr':>10}  per_task_sr")
for it, sr, ptsr in rows:
    sr_s = f"{sr:>10.3f}" if isinstance(sr, float) else f"{str(sr):>10}"
    print(f"{it:>12}  {sr_s}  {ptsr}")

with open(os.path.join(agg_dir, "all_iters_summary.json"), "w") as f:
    json.dump([{"iter": it, "overall_sr": sr, "per_task_sr": ptsr} for it, sr, ptsr in rows], f, indent=2)
print(f"\nsaved: {os.path.join(agg_dir, 'all_iters_summary.json')}")
PY

[ "${fail}" -eq 0 ] || { echo "(one or more shards failed; check logs)" >&2; exit 1; }
echo "Done."
