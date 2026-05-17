#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/Users/Max/miniforge3/envs/streamrl_arm/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
ALGO="${ALGO:-avg_default}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
SEEDS="${SEEDS:-1 2 3}"
LOG_DIR="${LOG_DIR:-logs}"
FORCE_RERUN="${FORCE_RERUN:-0}"

ACTOR_LR="${ACTOR_LR:-0.0003}"
CRITIC_LR="${CRITIC_LR:-0.0003}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.999}"
USE_ADAM_DEFAULT_BETAS="${USE_ADAM_DEFAULT_BETAS:-1}"
GAMMA="${GAMMA:-0.99}"
ALPHA_LR="${ALPHA_LR:-0.07}"
ACTION_DISTRIBUTION="${ACTION_DISTRIBUTION:-tanh_normal}"
TASKS="${TASKS:-mujoco|HalfCheetah-v4}"
COMPARE_TO_ALGO="${COMPARE_TO_ALGO:-avg_A}"
COMPARE_TAG="${COMPARE_TAG:-${ALGO}_vs_${COMPARE_TO_ALGO}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG_DIR"
RUN_LOG="${RUN_LOG:-$LOG_DIR/avg_default_phase2_$(date +%Y%m%d_%H%M%S).log}"

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
  local eval_csv="$RESULTS_ROOT/$backend/$ALGO/$safe_env/seed_$seed/eval.csv"

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
  local seed="$3"
  local beta_args=()

  if [ "$USE_ADAM_DEFAULT_BETAS" = "1" ]; then
    beta_args=(--use_adam_default_betas)
  else
    beta_args=(--beta1 "$BETA1" --beta2 "$BETA2")
  fi

  nice -n 10 "$PYTHON" -m algorithms.avg_baseline \
    --backend "$backend" \
    --env_name "$env_name" \
    --scaling_mode td_only \
    --algo "$ALGO" \
    --seed "$seed" \
    --total_steps "$TOTAL_STEPS" \
    --actor_lr "$ACTOR_LR" \
    --critic_lr "$CRITIC_LR" \
    "${beta_args[@]}" \
    --alpha_lr "$ALPHA_LR" \
    --gamma "$GAMMA" \
    --action_distribution "$ACTION_DISTRIBUTION" \
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
    --algo "$ALGO" \
    --env_name "$env_name" \
    --results_root "$RESULTS_ROOT"
}

compare_task() {
  local backend="$1"
  local env_name="$2"

  if [ -z "$COMPARE_TO_ALGO" ]; then
    return 0
  fi

  "$PYTHON" -m evaluation.compare_variants_plotter \
    --backend "$backend" \
    --env_name "$env_name" \
    --variants "$COMPARE_TO_ALGO=AVG paper hparams" "$ALGO=AVG Adam defaults" \
    --seeds $SEEDS \
    --output_tag "$COMPARE_TAG" \
    --results_root "$RESULTS_ROOT"
}

exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== AVG default Phase 2 batch started $(date) ==="
echo "Log: $RUN_LOG"
echo "Results path: $RESULTS_ROOT/<backend>/$ALGO/<env>/seed_<seed>"
echo "Vanilla AVG only: scaling_mode=td_only, use_layer_norm=false"
echo "Seeds: $SEEDS"
echo "Total steps: $TOTAL_STEPS"
echo "Eval interval: $EVAL_INTERVAL"
echo "Eval episodes: $EVAL_EPISODES"
if [ "$USE_ADAM_DEFAULT_BETAS" = "1" ]; then
  echo "Hyperparameters: actor_lr=$ACTOR_LR critic_lr=$CRITIC_LR betas=Adam defaults alpha_lr=$ALPHA_LR gamma=$GAMMA"
else
  echo "Hyperparameters: actor_lr=$ACTOR_LR critic_lr=$CRITIC_LR betas=($BETA1,$BETA2) alpha_lr=$ALPHA_LR gamma=$GAMMA"
fi
echo "Action distribution: $ACTION_DISTRIBUTION"
echo "Compare against: ${COMPARE_TO_ALGO:-disabled}"
echo "Force rerun: $FORCE_RERUN"

while IFS='|' read -r backend env_name; do
  [ -n "$backend" ] || continue

  echo
  echo "=== task $backend/$env_name started $(date) ==="

  for seed in $SEEDS; do
    if [ "$FORCE_RERUN" != "1" ] && run_is_complete "$backend" "$env_name" "$seed"; then
      echo "=== seed $seed already complete in $ALGO, skipping $(date) ==="
      continue
    fi

    echo "=== seed $seed started $(date) ==="
    run_seed "$backend" "$env_name" "$seed"
    echo "=== seed $seed finished $(date) ==="
    sleep 10
  done

  echo "=== plotting $backend/$env_name for $ALGO $(date) ==="
  plot_task "$backend" "$env_name"
  echo "=== comparing $ALGO against ${COMPARE_TO_ALGO:-disabled} for $backend/$env_name $(date) ==="
  compare_task "$backend" "$env_name"
  echo "=== task $backend/$env_name finished $(date) ==="
done <<< "$TASKS"

echo
echo "=== AVG default Phase 2 batch finished $(date) ==="
