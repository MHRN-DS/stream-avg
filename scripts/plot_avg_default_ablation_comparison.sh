#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/Users/Max/miniforge3/envs/streamrl_arm/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
BACKEND="${BACKEND:-mujoco}"
ENV_NAME="${ENV_NAME:-HalfCheetah-v4}"
SEEDS="${SEEDS:-1 2 3}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
OUTPUT_TAG="${OUTPUT_TAG:-avg_default_ablations_with_avg_A}"
WAIT_FOR_RESULTS="${WAIT_FOR_RESULTS:-0}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

VARIANTS=(
  "avg_A=AVG paper hparams vanilla"
  "avg_default=AVG default TD only"
  "avg_default_reward=AVG default + reward scaling"
  "avg_default_reward_entropy=AVG default + reward + eta scaling"
  "avg_default_ln=AVG default + LayerNorm"
  "avg_default_reward_ln=AVG default + reward scaling + LayerNorm"
  "avg_default_reward_entropy_ln=AVG default + reward + eta scaling + LayerNorm"
)

ALGOS=(
  "avg_A"
  "avg_default"
  "avg_default_reward"
  "avg_default_reward_entropy"
  "avg_default_ln"
  "avg_default_reward_ln"
  "avg_default_reward_entropy_ln"
)

sanitize_env_name() {
  local name="$1"
  name="${name//\//__}"
  name="${name//:/_}"
  printf '%s' "$name"
}

eval_is_complete() {
  local algo="$1"
  local seed="$2"
  local safe_env
  safe_env="$(sanitize_env_name "$ENV_NAME")"
  local eval_csv="$RESULTS_ROOT/$BACKEND/$algo/$safe_env/seed_$seed/eval.csv"

  if [ ! -f "$eval_csv" ]; then
    return 1
  fi

  local last_step
  last_step="$(awk -F, 'NR > 1 { step = $1 } END { print step + 0 }' "$eval_csv")"
  [ "$last_step" -ge "$TOTAL_STEPS" ]
}

all_results_complete() {
  local algo
  local seed

  for algo in "${ALGOS[@]}"; do
    for seed in $SEEDS; do
      eval_is_complete "$algo" "$seed" || return 1
    done
  done

  return 0
}

print_missing() {
  local algo
  local seed

  echo "Missing or incomplete eval.csv files:"
  for algo in "${ALGOS[@]}"; do
    for seed in $SEEDS; do
      if ! eval_is_complete "$algo" "$seed"; then
        echo "  $RESULTS_ROOT/$BACKEND/$algo/$(sanitize_env_name "$ENV_NAME")/seed_$seed/eval.csv"
      fi
    done
  done
}

if [ "$WAIT_FOR_RESULTS" = "1" ]; then
  until all_results_complete; do
    print_missing
    echo "Waiting ${WAIT_SECONDS}s before checking again..."
    sleep "$WAIT_SECONDS"
  done
else
  if ! all_results_complete; then
    print_missing
    exit 1
  fi
fi

"$PYTHON" -m evaluation.compare_variants_plotter \
  --backend "$BACKEND" \
  --env_name "$ENV_NAME" \
  --variants "${VARIANTS[@]}" \
  --seeds $SEEDS \
  --output_tag "$OUTPUT_TAG" \
  --results_root "$RESULTS_ROOT"
