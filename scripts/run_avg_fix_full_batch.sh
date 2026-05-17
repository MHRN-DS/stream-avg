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

TASKS="${TASKS:-mujoco|HalfCheetah-v4}"
RUN_HALFCHEETAH="${RUN_HALFCHEETAH:-1}"
RUN_FINGER_SPIN="${RUN_FINGER_SPIN:-1}"
HALFCHEETAH_COMPARE_TAG="${HALFCHEETAH_COMPARE_TAG:-avg_fix_full_halfcheetah}"
FINGER_COMPARE_TAG="${FINGER_COMPARE_TAG:-avg_fix_specific_finger_spin}"

HALFCHEETAH_VARIANTS="${HALFCHEETAH_VARIANTS:-$(cat <<'VARIANT_LIST'
avg_fix_specific_vanilla|td_only|0|0.0063|0.0087|0.0|0.999|0|0.07|0.99|tanh_normal|AVG fixed specific hparams vanilla
avg_fix_specific_reward|td_reward|0|0.0063|0.0087|0.0|0.999|0|0.07|0.99|tanh_normal|AVG fixed specific hparams + reward scaling
avg_fix_specific_reward_eta|td_reward_entropy|0|0.0063|0.0087|0.0|0.999|0|0.07|0.99|tanh_normal|AVG fixed specific hparams + reward + eta scaling
avg_fix_specific_ln|td_only|1|0.0063|0.0087|0.0|0.999|0|0.07|0.99|tanh_normal|AVG fixed specific hparams + LayerNorm
avg_fix_specific_reward_ln|td_reward|1|0.0063|0.0087|0.0|0.999|0|0.07|0.99|tanh_normal|AVG fixed specific hparams + reward scaling + LayerNorm
avg_fix_specific_reward_eta_ln|td_reward_entropy|1|0.0063|0.0087|0.0|0.999|0|0.07|0.99|tanh_normal|AVG fixed specific hparams + reward + eta scaling + LayerNorm
VARIANT_LIST
)}"

FINGER_SPIN_VARIANTS="${FINGER_SPIN_VARIANTS:-$(cat <<'VARIANT_LIST'
avg_fix_specific_tanh_vanilla|td_only|0|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|tanh_normal|AVG fixed specific hparams tanh-action vanilla
avg_fix_specific_tanh_reward|td_reward|0|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|tanh_normal|AVG fixed specific hparams tanh-action + reward scaling
avg_fix_specific_tanh_reward_eta|td_reward_entropy|0|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|tanh_normal|AVG fixed specific hparams tanh-action + reward + eta scaling
avg_fix_specific_tanh_ln|td_only|1|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|tanh_normal|AVG fixed specific hparams tanh-action + LayerNorm
avg_fix_specific_tanh_reward_ln|td_reward|1|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|tanh_normal|AVG fixed specific hparams tanh-action + reward scaling + LayerNorm
avg_fix_specific_tanh_reward_eta_ln|td_reward_entropy|1|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|tanh_normal|AVG fixed specific hparams tanh-action + reward + eta scaling + LayerNorm
avg_fix_specific_clip_vanilla|td_only|0|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|clipped_normal|AVG fixed specific hparams clipped-action vanilla
avg_fix_specific_clip_reward|td_reward|0|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|clipped_normal|AVG fixed specific hparams clipped-action + reward scaling
avg_fix_specific_clip_reward_eta|td_reward_entropy|0|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|clipped_normal|AVG fixed specific hparams clipped-action + reward + eta scaling
avg_fix_specific_clip_ln|td_only|1|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|clipped_normal|AVG fixed specific hparams clipped-action + LayerNorm
avg_fix_specific_clip_reward_ln|td_reward|1|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|clipped_normal|AVG fixed specific hparams clipped-action + reward scaling + LayerNorm
avg_fix_specific_clip_reward_eta_ln|td_reward_entropy|1|0.00038|8.7e-05|0.9|0.999|0|0.006|0.95|clipped_normal|AVG fixed specific hparams clipped-action + reward + eta scaling + LayerNorm
VARIANT_LIST
)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG_DIR"
RUN_LOG="${RUN_LOG:-$LOG_DIR/avg_fix_full_$(date +%Y%m%d_%H%M%S).log}"

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
  local action_distribution="${13}"
  local seed="${14}"
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
    --action_distribution "$action_distribution"
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

compare_variants() {
  local backend="$1"
  local env_name="$2"
  local variant_lines="$3"
  local output_tag="$4"
  local variant_specs=()

  while IFS='|' read -r algo scaling_mode use_layer_norm actor_lr critic_lr beta1 beta2 use_adam_default_betas alpha_lr gamma action_distribution label; do
    [ -n "$algo" ] || continue
    variant_specs+=("$algo=$label")
  done <<< "$variant_lines"

  "$PYTHON" -m evaluation.compare_variants_plotter \
    --backend "$backend" \
    --env_name "$env_name" \
    --variants "${variant_specs[@]}" \
    --seeds $SEEDS \
    --output_tag "$output_tag" \
    --results_root "$RESULTS_ROOT"
}

run_variant_set() {
  local backend="$1"
  local env_name="$2"
  local variant_lines="$3"
  local output_tag="$4"

  echo
  echo "=== task $backend/$env_name started $(date) ==="

  while IFS='|' read -r algo scaling_mode use_layer_norm actor_lr critic_lr beta1 beta2 use_adam_default_betas alpha_lr gamma action_distribution label; do
    [ -n "$algo" ] || continue

    echo
    echo "=== variant $algo started $(date) ==="
    echo "Label: $label"
    echo "Settings: scaling_mode=$scaling_mode use_layer_norm=$use_layer_norm action_distribution=$action_distribution"
    if [ "$use_adam_default_betas" = "1" ]; then
      echo "Hyperparameters: actor_lr=$actor_lr critic_lr=$critic_lr betas=Adam defaults alpha_lr=$alpha_lr gamma=$gamma"
    else
      echo "Hyperparameters: actor_lr=$actor_lr critic_lr=$critic_lr betas=($beta1,$beta2) alpha_lr=$alpha_lr gamma=$gamma"
    fi

    for seed in $SEEDS; do
      if [ "$FORCE_RERUN" != "1" ] && run_is_complete "$backend" "$algo" "$env_name" "$seed"; then
        echo "=== $algo seed $seed already complete, skipping $(date) ==="
        continue
      fi

      echo "=== $algo seed $seed started $(date) ==="
      run_seed "$backend" "$env_name" "$algo" "$scaling_mode" "$use_layer_norm" "$actor_lr" "$critic_lr" "$beta1" "$beta2" "$use_adam_default_betas" "$alpha_lr" "$gamma" "$action_distribution" "$seed"
      echo "=== $algo seed $seed finished $(date) ==="
      sleep 10
    done

    echo "=== plotting $algo on $backend/$env_name $(date) ==="
    plot_variant "$backend" "$env_name" "$algo"
    echo "=== variant $algo finished $(date) ==="
  done <<< "$variant_lines"

  echo "=== comparing variants on $backend/$env_name $(date) ==="
  compare_variants "$backend" "$env_name" "$variant_lines" "$output_tag"
  echo "=== task $backend/$env_name finished $(date) ==="
}

exec >> "$RUN_LOG" 2>&1

echo "=== AVG fixed full rerun started $(date) ==="
echo "Log: $RUN_LOG"
echo "Results path: $RESULTS_ROOT/<backend>/avg_fix*/<env>/seed_<seed>"
echo "Implementation flags: original AVG TD scaler, loss-unit scaler rewards, bootstrap through truncation, diagnostics enabled"
echo "Naming: avg_fix_specific_* uses tuned/paper hyperparameters."
echo "Seeds: $SEEDS"
echo "Total steps: $TOTAL_STEPS"
echo "Eval interval: $EVAL_INTERVAL"
echo "Eval episodes: $EVAL_EPISODES"
echo "Force rerun: $FORCE_RERUN"
echo
echo "HalfCheetah variants:"
printf '%s\n' "$HALFCHEETAH_VARIANTS"
echo
echo "Finger-spin variants:"
if [ "$RUN_FINGER_SPIN" = "1" ]; then
  printf '%s\n' "$FINGER_SPIN_VARIANTS"
else
  echo "disabled"
fi

if [ "$RUN_HALFCHEETAH" = "1" ]; then
  while IFS='|' read -r backend env_name; do
    [ -n "$backend" ] || continue
    run_variant_set "$backend" "$env_name" "$HALFCHEETAH_VARIANTS" "$HALFCHEETAH_COMPARE_TAG"
  done <<< "$TASKS"
else
  echo
  echo "=== HalfCheetah variants disabled for this invocation $(date) ==="
fi

if [ "$RUN_FINGER_SPIN" = "1" ]; then
  run_variant_set "dmcontrol" "finger-spin-v0" "$FINGER_SPIN_VARIANTS" "$FINGER_COMPARE_TAG"
fi

echo
echo "=== AVG fixed full rerun finished $(date) ==="
