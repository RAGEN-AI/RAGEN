#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_selected_envs.sh --model-path PATH [options] [env ...]

Runs evaluation sequentially for the selected environments.
If no env is provided, it runs:
  sokoban frozenlake deepcoder bandit countdown metamathqa

Options:
  --model-path PATH        Hugging Face model id or local checkpoint path
  --gpus LIST              GPU list passed to Hydra and CUDA_VISIBLE_DEVICES
                           default: 0
  --output-root DIR        Root directory for eval outputs and logs
                           default: results/eval_multi
  --quick                  Smaller validation batches for smoke tests
  --override KEY=VALUE     Extra Hydra override, can be repeated
  -h, --help               Show this help

Examples:
  scripts/eval_selected_envs.sh \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --gpus 0

  scripts/eval_selected_envs.sh \
    --model-path /path/to/checkpoint \
    --gpus 0,1 \
    --quick \
    frozenlake deepcoder
EOF
}

normalize_env_name() {
  local raw="${1,,}"
  raw="${raw//-/_}"
  case "$raw" in
    sokoban) echo "sokoban" ;;
    frozenlake|frozen_lake) echo "frozenlake" ;;
    deepcoder) echo "deepcoder" ;;
    bandit) echo "bandit" ;;
    countdown) echo "countdown" ;;
    metamathqa|meta_math_qa|meta_mathqa) echo "metamathqa" ;;
    lightsout|lights_out|lights-out) echo "lightsout" ;;
    *)
      echo "Unknown env: $1" >&2
      echo "Available envs: sokoban frozenlake deepcoder bandit countdown metamathqa lightsout" >&2
      exit 2
      ;;
  esac
}

declare -A CONFIG_MAP=(
  [sokoban]="_2_sokoban"
  [frozenlake]="_3_frozen_lake"
  [deepcoder]="_10_deepcoder"
  [bandit]="_1_bandit"
  [countdown]="_4_countdown"
  [metamathqa]="_5_metamathqa"
  [lightsout]="_11_lights_out"
)

MODEL_PATH=""
GPU_LIST="0"
OUTPUT_ROOT="results/eval_multi"
QUICK_MODE=0
PYTHON_BIN="${PYTHON_BIN:-python}"
declare -a EXTRA_OVERRIDES=()
declare -a REQUESTED_ENVS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --quick)
      QUICK_MODE=1
      shift
      ;;
    --override)
      EXTRA_OVERRIDES+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      REQUESTED_ENVS+=("$(normalize_env_name "$1")")
      shift
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "--model-path is required" >&2
  usage
  exit 2
fi

if [[ ${#REQUESTED_ENVS[@]} -eq 0 ]]; then
  REQUESTED_ENVS=(sokoban frozenlake deepcoder bandit countdown metamathqa)
fi

mkdir -p "$OUTPUT_ROOT/logs"

declare -A STATUS_MAP=()
declare -a FAILED_ENVS=()

for env_name in "${REQUESTED_ENVS[@]}"; do
  config_name="${CONFIG_MAP[$env_name]}"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_path="$OUTPUT_ROOT/logs/${timestamp}_${env_name}.log"

  cmd=(
    "$PYTHON_BIN" -m ragen.llm_agent.agent_proxy
    --config-name "$config_name"
    "model_path=${MODEL_PATH}"
    "trainer.experiment_name=${env_name}"
    "trainer.local_log_dir=${OUTPUT_ROOT}"
    "system.CUDA_VISIBLE_DEVICES='${GPU_LIST}'"
  )

  if [[ "$QUICK_MODE" -eq 1 ]]; then
    case "$env_name" in
      bandit)
        cmd+=(
          "es_manager.val.env_groups=32"
          "es_manager.val.group_size=1"
          "es_manager.val.env_configs.n_groups=[16,16]"
        )
        ;;
      *)
        cmd+=(
          "es_manager.val.env_groups=32"
          "es_manager.val.group_size=1"
          "es_manager.val.env_configs.n_groups=[32]"
        )
        ;;
    esac
  fi

  if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_OVERRIDES[@]}")
  fi

  echo "============================================================"
  echo "[$env_name] config=${config_name}"
  echo "[$env_name] log=${log_path}"
  printf '[%s] cmd:' "$env_name"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if CUDA_VISIBLE_DEVICES="$GPU_LIST" "${cmd[@]}" 2>&1 | tee "$log_path"; then
    STATUS_MAP["$env_name"]="ok"
  else
    STATUS_MAP["$env_name"]="failed"
    FAILED_ENVS+=("$env_name")
  fi
done

echo "============================================================"
echo "Evaluation summary:"
for env_name in "${REQUESTED_ENVS[@]}"; do
  echo "  ${env_name}: ${STATUS_MAP[$env_name]}"
done

if [[ ${#FAILED_ENVS[@]} -gt 0 ]]; then
  echo "Failed envs: ${FAILED_ENVS[*]}" >&2
  exit 1
fi
