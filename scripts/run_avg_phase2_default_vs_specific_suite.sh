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
RUN_COMPARISONS="${RUN_COMPARISONS:-0}"

DEFAULT_ACTOR_LR="${DEFAULT_ACTOR_LR:-0.0003}"
DEFAULT_CRITIC_LR="${DEFAULT_CRITIC_LR:-0.0003}"
DEFAULT_ALPHA_LR="${DEFAULT_ALPHA_LR:-0.07}"
DEFAULT_GAMMA="${DEFAULT_GAMMA:-0.99}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG_DIR"
RUN_LOG="${RUN_LOG:-$LOG_DIR/avg_phase2_default_vs_specific_suite_$(date +%Y%m%d_%H%M%S).log}"

TASKS="${TASKS:-$(cat <<'TASK_LIST'
mujoco|Hopper-v4|1.1e-05|7.7e-05|0.0|0.999|0.3|0.99
mujoco|Walker2d-v4|1.1e-05|7.7e-05|0.0|0.999|0.3|0.99
mujoco|Ant-v4|0.0063|0.0087|0.0|0.999|0.07|0.99
dmcontrol|cheetah-run|0.00038|8.7e-05|0.9|0.999|0.006|0.95
dmcontrol|walker-walk|0.00038|8.7e-05|0.9|0.999|0.006|0.95
dmcontrol|dog-walk|6e-06|8e-05|0.0|0.999|0.009|0.95
TASK_LIST
)}"

VARIANTS="${VARIANTS:-$(cat <<'VARIANT_LIST'
vanilla|td_only|0|vanilla
reward|td_reward|0|reward scaling
reward_eta|td_reward_entropy|0|reward + eta scaling
ln|td_only|1|LayerNorm
reward_ln|td_reward|1|reward scaling + LayerNorm
reward_eta_ln|td_reward_entropy|1|reward + eta scaling + LayerNorm
VARIANT_LIST
)}"

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
  local actor_lr="$6"
  local critic_lr="$7"
  local beta1="$8"
  local beta2="$9"
  local use_adam_default_betas="${10}"
  local alpha_lr="${11}"
  local gamma="${12}"
  local seed="${13}"
  local beta_args=()
  local cmd=()

  if [ "$use_adam_default_betas" = "1" ]; then
    beta_args=(--use_adam_default_betas)
  else
    beta_args=(--beta1 "$beta1" --beta2 "$beta2")
  fi

  cmd=(nice -n 10 "$PYTHON" -m algorithms.avg_baseline
    --backend "$backend"
    --env_name "$env_name"
    --scaling_mode "$scaling_mode"
    --algo "$algo"
    --td_scaler_impl original
    --td_scaler_reward_units loss
    --bootstrap_on_truncation
    --log_update_stats)

  if [ "$use_layer_norm" = "1" ]; then
    cmd+=(--use_layer_norm)
  fi

  cmd+=(
    --seed "$seed"
    --total_steps "$TOTAL_STEPS"
    --actor_lr "$actor_lr"
    --critic_lr "$critic_lr"
    "${beta_args[@]}"
    --alpha_lr "$alpha_lr"
    --gamma "$gamma"
    --action_distribution tanh_normal
    --eval_interval "$EVAL_INTERVAL"
    --eval_episodes "$EVAL_EPISODES"
    --results_root "$RESULTS_ROOT"
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

compare_split() {
  local backend="$1"
  local env_name="$2"
  local tag="$3"
  local variant_specs=()

  if [ "$tag" = "no_ln" ]; then
    variant_specs=(
      "avg_fix_default_vanilla=default vanilla"
      "avg_fix_default_reward=default reward"
      "avg_fix_default_reward_eta=default reward+eta"
      "avg_fix_specific_vanilla=specific vanilla"
      "avg_fix_specific_reward=specific reward"
      "avg_fix_specific_reward_eta=specific reward+eta"
    )
  else
    variant_specs=(
      "avg_fix_default_ln=default LN"
      "avg_fix_default_reward_ln=default reward+LN"
      "avg_fix_default_reward_eta_ln=default reward+eta+LN"
      "avg_fix_specific_ln=specific LN"
      "avg_fix_specific_reward_ln=specific reward+LN"
      "avg_fix_specific_reward_eta_ln=specific reward+eta+LN"
    )
  fi

  "$PYTHON" -m evaluation.compare_variants_plotter \
    --backend "$backend" \
    --env_name "$env_name" \
    --variants "${variant_specs[@]}" \
    --seeds $SEEDS \
    --output_tag "avg_phase2_default_vs_specific_${tag}" \
    --results_root "$RESULTS_ROOT"
}

run_task_set() {
  local backend="$1"
  local env_name="$2"
  local specific_actor_lr="$3"
  local specific_critic_lr="$4"
  local specific_beta1="$5"
  local specific_beta2="$6"
  local specific_alpha_lr="$7"
  local specific_gamma="$8"

  echo
  echo "=== task $backend/$env_name started $(date) ==="
  echo "Default hyperparameters: actor_lr=$DEFAULT_ACTOR_LR critic_lr=$DEFAULT_CRITIC_LR betas=Adam defaults alpha_lr=$DEFAULT_ALPHA_LR gamma=$DEFAULT_GAMMA"
  echo "Paper-specific hyperparameters: actor_lr=$specific_actor_lr critic_lr=$specific_critic_lr betas=($specific_beta1,$specific_beta2) alpha_lr=$specific_alpha_lr gamma=$specific_gamma"

  for family in default specific; do
    if [ "$family" = "default" ]; then
      actor_lr="$DEFAULT_ACTOR_LR"
      critic_lr="$DEFAULT_CRITIC_LR"
      beta1="0.9"
      beta2="0.999"
      use_adam_default_betas="1"
      alpha_lr="$DEFAULT_ALPHA_LR"
      gamma="$DEFAULT_GAMMA"
    else
      actor_lr="$specific_actor_lr"
      critic_lr="$specific_critic_lr"
      beta1="$specific_beta1"
      beta2="$specific_beta2"
      use_adam_default_betas="0"
      alpha_lr="$specific_alpha_lr"
      gamma="$specific_gamma"
    fi

    echo
    echo "=== family $family on $backend/$env_name started $(date) ==="

    while IFS='|' read -r suffix scaling_mode use_layer_norm label; do
      [ -n "$suffix" ] || continue
      local algo="avg_fix_${family}_${suffix}"

      echo
      echo "=== variant $algo started $(date) ==="
      echo "Label: $family $label"
      echo "Settings: scaling_mode=$scaling_mode use_layer_norm=$use_layer_norm"

      for seed in $SEEDS; do
        if [ "$FORCE_RERUN" != "1" ] && run_is_complete "$backend" "$algo" "$env_name" "$seed"; then
          echo "=== $algo seed $seed already complete, skipping $(date) ==="
          continue
        fi

        echo "=== $algo seed $seed started $(date) ==="
        run_seed "$backend" "$env_name" "$algo" "$scaling_mode" "$use_layer_norm" "$actor_lr" "$critic_lr" "$beta1" "$beta2" "$use_adam_default_betas" "$alpha_lr" "$gamma" "$seed"
        echo "=== $algo seed $seed finished $(date) ==="
        sleep 10
      done

      echo "=== plotting $algo on $backend/$env_name $(date) ==="
      plot_variant "$backend" "$env_name" "$algo"
      echo "=== variant $algo finished $(date) ==="
    done <<< "$VARIANTS"

    echo "=== family $family on $backend/$env_name finished $(date) ==="
  done

  if [ "$RUN_COMPARISONS" = "1" ]; then
    echo "=== comparing no-LN variants on $backend/$env_name $(date) ==="
    compare_split "$backend" "$env_name" "no_ln"
    echo "=== comparing LN variants on $backend/$env_name $(date) ==="
    compare_split "$backend" "$env_name" "ln"
  else
    echo "=== comparison plots disabled for $backend/$env_name $(date) ==="
  fi
  echo "=== task $backend/$env_name finished $(date) ==="
}

exec >> "$RUN_LOG" 2>&1

echo "=== AVG Phase 2 default-vs-specific ablation suite started $(date) ==="
echo "Log: $RUN_LOG"
echo "Results path: $RESULTS_ROOT/<backend>/avg_fix_{default,specific}_*/<env>/seed_<seed>"
echo "Implementation flags: original AVG TD scaler, loss-unit scaler rewards, bootstrap through truncation, diagnostics enabled"
echo "Action distribution: tanh_normal"
echo "Seeds: $SEEDS"
echo "Total steps: $TOTAL_STEPS"
echo "Eval interval: $EVAL_INTERVAL"
echo "Eval episodes: $EVAL_EPISODES"
echo "Force rerun: $FORCE_RERUN"
echo "Run comparisons: $RUN_COMPARISONS"
echo
echo "Tasks:"
printf '%s\n' "$TASKS"
echo
echo "Variants:"
printf '%s\n' "$VARIANTS"

while IFS='|' read -r backend env_name actor_lr critic_lr beta1 beta2 alpha_lr gamma; do
  [ -n "$backend" ] || continue
  run_task_set "$backend" "$env_name" "$actor_lr" "$critic_lr" "$beta1" "$beta2" "$alpha_lr" "$gamma"
done <<< "$TASKS"

echo
echo "=== AVG Phase 2 default-vs-specific ablation suite finished $(date) ==="
