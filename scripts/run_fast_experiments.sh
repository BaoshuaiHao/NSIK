#!/usr/bin/env bash
set -euo pipefail

# Fast, resumable NSIK experiments.
#
# Usage examples:
#   bash scripts/run_fast_experiments.sh
#   DATASETS="WN18RR_v1 fb237_v1 nell_v1" EPOCHS=20 bash scripts/run_fast_experiments.sh
#   ROLE_DIMS="4 8 12" VARIANTS="full no_global no_attn" bash scripts/run_fast_experiments.sh
#
# The script writes a compact CSV summary to experiments/reviewer_results.csv.

DATASETS=${DATASETS:-"WN18RR_v1"}
ROLE_DIMS=${ROLE_DIMS:-"8"}
VARIANTS=${VARIANTS:-"full no_global no_attn"}
EPOCHS=${EPOCHS:-20}
SEEDS=${SEEDS:-"0"}
LR=${LR:-0.0005}
HOP=${HOP:-3}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LINKS=${MAX_LINKS:-10000}
NUM_WORKERS=${NUM_WORKERS:-4}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python3}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-5}
TOP_GLOBAL=${TOP_GLOBAL:-100}
PER_ROLE=${PER_ROLE:-10}
LAMBDA_RATIO=${LAMBDA_RATIO:-0.5}
PREPROCESS=${PREPROCESS:-1}
RUN_RANKING=${RUN_RANKING:-0}
RESULT_CSV=${RESULT_CSV:-"experiments/reviewer_results.csv"}

mkdir -p experiments
if [ ! -f "$RESULT_CSV" ]; then
  printf "timestamp,dataset,test_dataset,role_dim,variant,seed,diffusion_steps,top_global,per_role,lambda_ratio,experiment,auc,auc_pr,hits10\n" > "$RESULT_CSV"
fi

test_dataset_for() {
  printf "%s_ind" "$1"
}

variant_flags() {
  case "$1" in
    full)
      printf ""
      ;;
    no_global)
      printf "%s" "--disable_global_emb"
      ;;
    no_roles)
      printf "%s" "--disable_node_roles"
      ;;
    no_attn)
      printf "%s" "--disable_attn"
      ;;
    no_global_no_attn)
      printf "%s" "--disable_global_emb --disable_attn"
      ;;
    *)
      printf "Unknown variant: %s\n" "$1" >&2
      return 1
      ;;
  esac
}

eval_variant_flags() {
  case "$1" in
    full|no_attn)
      printf ""
      ;;
    no_global|no_global_no_attn)
      printf "%s" "--disable_global_emb"
      ;;
    no_roles)
      printf "%s" "--disable_node_roles"
      ;;
    *)
      printf "Unknown variant: %s\n" "$1" >&2
      return 1
      ;;
  esac
}

extract_auc_metrics() {
  local log_path=$1
  "$PYTHON" - "$log_path" <<'PY'
import ast
import re
import sys

path = sys.argv[1]
auc = ""
auc_pr = ""
with open(path, "r") as f:
    text = f.read()
matches = re.findall(r"Test Set Performance:(\{[^}]+\})", text)
if matches:
    metrics = ast.literal_eval(matches[-1])
    auc = metrics.get("auc", "")
    auc_pr = metrics.get("auc_pr", "")
print(f"{auc},{auc_pr}")
PY
}

extract_hits10() {
  local log_path=$1
  if [ ! -f "$log_path" ]; then
    printf ""
    return
  fi
  "$PYTHON" - "$log_path" <<'PY'
import re
import sys

text = open(sys.argv[1], "r").read()
matches = re.findall(r"MRR \| Hits@1 \| Hits@5 \| Hits@10 : [^|]+ \| [^|]+ \| [^|]+ \| ([0-9.eE+-]+)", text)
print(matches[-1] if matches else "")
PY
}

for role_dim in $ROLE_DIMS; do
  if [ "$PREPROCESS" = "1" ]; then
    preprocess_datasets=""
    for dataset in $DATASETS; do
      test_dataset=$(test_dataset_for "$dataset")
      preprocess_datasets="$preprocess_datasets $dataset $test_dataset"
    done

    "$PYTHON" compute_roles.py --datasets $preprocess_datasets
    "$PYTHON" build_rR_matrix.py --k-roles "$role_dim" --datasets $preprocess_datasets
    "$PYTHON" compute_global_diffusion.py \
      --datasets $preprocess_datasets \
      --diffusion-steps "$DIFFUSION_STEPS" \
      --top-global "$TOP_GLOBAL" \
      --per-role "$PER_ROLE" \
      --lambda-ratio "$LAMBDA_RATIO"
  fi

  for dataset in $DATASETS; do
    test_dataset=$(test_dataset_for "$dataset")

    for variant in $VARIANTS; do
      flags=$(variant_flags "$variant")
      eval_flags=$(eval_variant_flags "$variant")
      for seed in $SEEDS; do
        exp="fast_${dataset}_r${role_dim}_${variant}_e${EPOCHS}_s${seed}_d${DIFFUSION_STEPS}_k${TOP_GLOBAL}_p${PER_ROLE}_l${LAMBDA_RATIO}"

        echo "=== Train: dataset=$dataset role_dim=$role_dim variant=$variant seed=$seed exp=$exp ==="
        "$PYTHON" train.py \
          -d "$dataset" \
          -e "$exp" \
          --num_epochs "$EPOCHS" \
          --eval_every 5 \
          --early_stop 10 \
          --lr "$LR" \
          --hop "$HOP" \
          --batch_size "$BATCH_SIZE" \
          --max_links "$MAX_LINKS" \
          --num_workers "$NUM_WORKERS" \
          --gpu "$GPU" \
          --seed "$seed" \
          $flags

        echo "=== Test AUC: dataset=$test_dataset exp=$exp ==="
        "$PYTHON" test_auc.py \
          -d "$test_dataset" \
          -e "$exp" \
          --hop "$HOP" \
          --batch_size "$BATCH_SIZE" \
          --max_links "$MAX_LINKS" \
          --num_workers "$NUM_WORKERS" \
          --gpu "$GPU" \
          $eval_flags

        test_log="experiments/${exp}/test_${test_dataset}_0/log_test.txt"
        metrics=$(extract_auc_metrics "$test_log")
        hits10=""

        if [ "$RUN_RANKING" = "1" ]; then
          echo "=== Test ranking: dataset=$test_dataset exp=$exp ==="
          ranking_flags=""
          if [[ "$variant" == *"no_global"* ]]; then
            ranking_flags="--disable_global_emb"
          fi
          "$PYTHON" test_ranking.py \
            -d "$test_dataset" \
            -e "$exp" \
            --hop "$HOP" \
            $ranking_flags
          rank_log=$(ls -t "experiments/${exp}"/log_rank_test_*.txt | head -n 1)
          hits10=$(extract_hits10 "$rank_log")
        fi

        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
          "$(date '+%Y-%m-%d %H:%M:%S')" \
          "$dataset" \
          "$test_dataset" \
          "$role_dim" \
          "$variant" \
          "$seed" \
          "$DIFFUSION_STEPS" \
          "$TOP_GLOBAL" \
          "$PER_ROLE" \
          "$LAMBDA_RATIO" \
          "$exp" \
          "$metrics" \
          "$hits10" >> "$RESULT_CSV"
      done
    done
  done
done

echo "Done. Results: $RESULT_CSV"
