#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/Users/Max/miniforge3/envs/streamrl_arm/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
SEEDS="${SEEDS:-1 2 3}"
LOG_DIR="${LOG_DIR:-logs}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG_DIR"
RUN_LOG="${RUN_LOG:-$LOG_DIR/avg_A_paper_mujoco_dmcontrol_$(date +%Y%m%d_%H%M%S).log}"

sanitize_env_name() {
  local name="$1"
  name="${name//\//__}"
  name="${name//:/_}"
  printf '%s' "$name"
}

run_is_complete() {
  local backend="$1"
  local env_name="$2"
  local seed="$3"
  local safe_env
  safe_env="$(sanitize_env_name "$env_name")"
  local eval_csv="$RESULTS_ROOT/$backend/avg_A/$safe_env/seed_$seed/eval.csv"

  if [ ! -f "$eval_csv" ]; then
    return 1
  fi

  local last_step
  last_step="$(awk -F, 'NR > 1 { step = $1 } END { print step + 0 }' "$eval_csv")"
  [ "$last_step" -ge "$TOTAL_STEPS" ]
}

run_seed() {
  local backend="$1"
  local env_name="$2"
  local actor_lr="$3"
  local critic_lr="$4"
  local beta1="$5"
  local alpha_lr="$6"
  local gamma="$7"
  local seed="$8"

  nice -n 10 "$PYTHON" -m algorithms.avg_baseline \
    --backend "$backend" \
    --env_name "$env_name" \
    --scaling_mode td_only \
    --seed "$seed" \
    --total_steps "$TOTAL_STEPS" \
    --actor_lr "$actor_lr" \
    --critic_lr "$critic_lr" \
    --beta1 "$beta1" \
    --alpha_lr "$alpha_lr" \
    --gamma "$gamma" \
    --eval_interval "$EVAL_INTERVAL" \
    --eval_episodes "$EVAL_EPISODES" \
    --results_root "$RESULTS_ROOT" \
    --device cpu
}

plot_task() {
  local backend="$1"
  local env_name="$2"

  "$PYTHON" -m evaluation.fixed_plot_results \
    --backend "$backend" \
    --algo avg_A \
    --env_name "$env_name" \
    --results_root "$RESULTS_ROOT"
}

TASKS=$(
  cat <<'TASK_LIST'
mujoco|Hopper-v4|1.1e-05|7.7e-05|0.0|0.3|0.99
mujoco|Walker2d-v4|1.1e-05|7.7e-05|0.0|0.3|0.99
mujoco|Ant-v4|0.0063|0.0087|0.0|0.07|0.99
dmcontrol|cheetah-run|0.00038|8.7e-05|0.9|0.006|0.95
dmcontrol|dog-walk|6e-06|8e-05|0.0|0.009|0.95
dmcontrol|dog-stand|6e-06|8e-05|0.0|0.009|0.95
TASK_LIST
)

exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== AVG-A paper batch started $(date) ==="
echo "Log: $RUN_LOG"
echo "Results root: $RESULTS_ROOT"
echo "Seeds: $SEEDS"
echo "Total steps: $TOTAL_STEPS"
echo "Eval interval: $EVAL_INTERVAL"
echo "Eval episodes: $EVAL_EPISODES"
echo "Force rerun: $FORCE_RERUN"

while IFS='|' read -r backend env_name actor_lr critic_lr beta1 alpha_lr gamma; do
  [ -n "$backend" ] || continue

  echo
  echo "=== task $backend/$env_name started $(date) ==="
  echo "Hyperparameters: actor_lr=$actor_lr critic_lr=$critic_lr beta1=$beta1 beta2=0.999 alpha_lr=$alpha_lr gamma=$gamma"

  for seed in $SEEDS; do
    if [ "$FORCE_RERUN" != "1" ] && run_is_complete "$backend" "$env_name" "$seed"; then
      echo "=== seed $seed already complete, skipping $(date) ==="
      continue
    fi

    echo "=== seed $seed started $(date) ==="
    run_seed "$backend" "$env_name" "$actor_lr" "$critic_lr" "$beta1" "$alpha_lr" "$gamma" "$seed"
    echo "=== seed $seed finished $(date) ==="
    sleep 10
  done

  echo "=== plotting $backend/$env_name $(date) ==="
  plot_task "$backend" "$env_name"
  echo "=== task $backend/$env_name finished $(date) ==="
done <<< "$TASKS"

echo
echo "=== AVG-A paper batch finished $(date) ==="
