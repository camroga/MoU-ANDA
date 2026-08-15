#!/bin/bash

set -euo pipefail

export PATH="/opt/conda/bin:$PATH"

PROJECT_DIR="/workspace/thesis/baseline/Training_ANDA"

OBAMA_DATASET="/workspace/thesis/100-shot-obama.zip"
PANDA_DATASET="/workspace/thesis/100-shot-panda.zip"

OUTDIR_ROOT="/workspace/thesis/baseline/runs/anda-original"
LOGDIR="/workspace/thesis/baseline/logs"

DATASETS=(
    "$OBAMA_DATASET"
    "$PANDA_DATASET"
)

DATASET_NAMES=(
    "obama"
    "panda"
)

SEEDS=(2 3 4)

TRAIN_KIMG=500

mkdir -p "$OUTDIR_ROOT" "$LOGDIR"

cd "$PROJECT_DIR"

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
GIT_COMMIT="$(git rev-parse --short HEAD)"

echo "===== GPU CHECK ====="
nvidia-smi
echo "====================="

PIDS=()
TASK_INDEX=0

for DATASET_INDEX in "${!DATASETS[@]}"; do
    for SEED_INDEX in "${!SEEDS[@]}"; do

        GPU_ID="$TASK_INDEX"

        DATASET="${DATASETS[$DATASET_INDEX]}"
        DATASET_NAME="${DATASET_NAMES[$DATASET_INDEX]}"
        SEED="${SEEDS[$SEED_INDEX]}"

        EXPERIMENT_NAME="anda-original-${DATASET_NAME}-seed-${SEED}"
        OUTDIR="${OUTDIR_ROOT}/${EXPERIMENT_NAME}"
        LOGFILE="${LOGDIR}/${EXPERIMENT_NAME}.log"

        mkdir -p "$OUTDIR"

        cat > "${OUTDIR}/submitted_configuration.txt" <<EOF
experiment=${EXPERIMENT_NAME}
method=original_anda
gpu=${GPU_ID}
dataset=${DATASET_NAME}
dataset_path=${DATASET}
training_kimg=${TRAIN_KIMG}
seed=${SEED}
git_branch=${GIT_BRANCH}
git_commit=${GIT_COMMIT}
EOF

        echo "Launching ${EXPERIMENT_NAME} on GPU ${GPU_ID}"

        (
            export CUDA_VISIBLE_DEVICES="$GPU_ID"

            python -u train.py \
              --outdir="$OUTDIR" \
              --data="$DATASET" \
              --cfg=low_shot \
              --mirror=true \
              --gpus=1 \
              --seed="$SEED" \
              --kimg="$TRAIN_KIMG" \
              --snap=10 \
              --metrics=fid50k_full

        ) > "$LOGFILE" 2>&1 &

        PIDS+=("$!")
        TASK_INDEX=$((TASK_INDEX + 1))

    done
done

STATUS=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        STATUS=1
    fi
done

if [ "$STATUS" -eq 0 ]; then
    echo "All baseline experiments completed successfully."
    echo "Stopping RunPod..."
    runpodctl pod stop "$RUNPOD_POD_ID" || true
else
    echo "One or more baseline experiments failed."
    echo "RunPod will remain running for debugging."
fi

exit "$STATUS"