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

ACTOR_LR="${ACTOR_LR:-0.0003}"
CRITIC_LR="${CRITIC_LR:-0.0003}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.999}"
USE_ADAM_DEFAULT_BETAS="${USE_ADAM_DEFAULT_BETAS:-1}"
GAMMA="${GAMMA:-0.99}"
ALPHA_LR="${ALPHA_LR:-0.07}"
ACTION_DISTRIBUTION="${ACTION_DISTRIBUTION:-tanh_normal}"
TASKS="${TASKS:-mujoco|HalfCheetah-v4}"
COMPARE_TAG="${COMPARE_TAG:-avg_fix_ablations_with_avg_default_and_avg_A}"

VARIANTS="${VARIANTS:-$(cat <<'VARIANT_LIST'
avg_fix_td|td_only|0|AVG fixed impl TD only
avg_fix_reward|td_reward|0|AVG fixed impl + reward scaling
avg_fix_reward_eta|td_reward_entropy|0|AVG fixed impl + reward + eta scaling
avg_fix_ln|td_only|1|AVG fixed impl + LayerNorm
avg_fix_reward_ln|td_reward|1|AVG fixed impl + reward scaling + LayerNorm
avg_fix_reward_eta_ln|td_reward_entropy|1|AVG fixed impl + reward + eta scaling + LayerNorm
VARIANT_LIST
)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG_DIR"
RUN_LOG="${RUN_LOG:-$LOG_DIR/avg_fix_ablations_halfcheetah_1m_s1s2s3.log}"

sanitize_env_name() {
  local name="$1"
  name="${name//\//__}"
  name="${name//:/_}"
  printf '%s' "$name"
}

run_is_complete() {
  local backend="$1"
  local algo="$2"
  local env_name="$3"
  local seed="$4"
  local safe_env
  safe_env="$(sanitize_env_name "$env_name")"
  local eval_csv="$RESULTS_ROOT/$backend/$algo/$safe_env/seed_$seed/eval.csv"

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
  local algo="$3"
  local scaling_mode="$4"
  local use_layer_norm="$5"
  local seed="$6"
  local beta_args=()
  local cmd=()

  if [ "$USE_ADAM_DEFAULT_BETAS" = "1" ]; then
    beta_args=(--use_adam_default_betas)
  else
    beta_args=(--beta1 "$BETA1" --beta2 "$BETA2")
  fi

  cmd=(nice -n 10 "$PYTHON" -m algorithms.avg_baseline \
    --backend "$backend" \
    --env_name "$env_name" \
    --scaling_mode "$scaling_mode" \
    --algo "$algo" \
    --td_scaler_impl original \
    --td_scaler_reward_units loss \
    --bootstrap_on_truncation \
    --log_update_stats)

  if [ "$use_layer_norm" = "1" ]; then
    cmd+=(--use_layer_norm)
  fi

  cmd+=( \
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
    --device cpu)

  "${cmd[@]}"
}

plot_variant() {
  local backend="$1"
  local env_name="$2"
  local algo="$3"

  "$PYTHON" -m evaluation.fixed_plot_results \
    --backend "$backend" \
    --algo "$algo" \
    --env_name "$env_name" \
    --results_root "$RESULTS_ROOT"
}

compare_variants() {
  local backend="$1"
  local env_name="$2"
  local variant_specs=(
    "avg_A=AVG paper hparams vanilla"
    "avg_default=AVG default hparams vanilla"
  )

  while IFS='|' read -r algo scaling_mode use_layer_norm label; do
    [ -n "$algo" ] || continue
    variant_specs+=("$algo=$label")
  done <<< "$VARIANTS"

  "$PYTHON" -m evaluation.compare_variants_plotter \
    --backend "$backend" \
    --env_name "$env_name" \
    --variants "${variant_specs[@]}" \
    --seeds $SEEDS \
    --output_tag "$COMPARE_TAG" \
    --results_root "$RESULTS_ROOT"
}

exec >> "$RUN_LOG" 2>&1

echo "=== AVG fixed ablation batch started $(date) ==="
echo "Log: $RUN_LOG"
echo "Results path: $RESULTS_ROOT/<backend>/avg_fix*/<env>/seed_<seed>"
echo "Naming: avg_default is reserved for the vanilla default-hyperparameter search."
echo "Fixed implementation ablations: original AVG TD scaler, loss-unit TD stats, bootstrap through truncation, diagnostics enabled"
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
echo "Force rerun: $FORCE_RERUN"
echo "Variants:"
printf '%s\n' "$VARIANTS"

while IFS='|' read -r backend env_name; do
  [ -n "$backend" ] || continue

  echo
  echo "=== task $backend/$env_name started $(date) ==="

  while IFS='|' read -r algo scaling_mode use_layer_norm label; do
    [ -n "$algo" ] || continue

    echo
    echo "=== variant $algo started $(date) ==="
    echo "Label: $label"
    echo "Settings: scaling_mode=$scaling_mode use_layer_norm=$use_layer_norm"

    for seed in $SEEDS; do
      if [ "$FORCE_RERUN" != "1" ] && run_is_complete "$backend" "$algo" "$env_name" "$seed"; then
        echo "=== $algo seed $seed already complete, skipping $(date) ==="
        continue
      fi

      echo "=== $algo seed $seed started $(date) ==="
      run_seed "$backend" "$env_name" "$algo" "$scaling_mode" "$use_layer_norm" "$seed"
      echo "=== $algo seed $seed finished $(date) ==="
      sleep 10
    done

    echo "=== plotting $algo on $backend/$env_name $(date) ==="
    plot_variant "$backend" "$env_name" "$algo"
    echo "=== variant $algo finished $(date) ==="
  done <<< "$VARIANTS"

  echo "=== comparing fixed ablations on $backend/$env_name $(date) ==="
  compare_variants "$backend" "$env_name"
  echo "=== task $backend/$env_name finished $(date) ==="
done <<< "$TASKS"

echo
echo "=== AVG fixed ablation batch finished $(date) ==="
