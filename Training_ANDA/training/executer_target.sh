#!/bin/bash

set -euo pipefail

PROJECT_DIR="/workspace/thesis/calibration"
DATASET="/workspace/thesis/obama.zip"
OUTDIR_ROOT="/workspace/thesis/calibration/runs/mou-target-recalibration"
LOGDIR="/workspace/thesis/calibration/logs"

# Values to evaluate.
ANDA_TARGETS=(0.20 0.40 0.60)

# Fixed values.
ANDA_INTERVAL=4
ANDA_KIMG=400
MOU_EPSILON=0.20
SEED=1
TRAIN_KIMG=400

mkdir -p "$OUTDIR_ROOT" "$LOGDIR"

cd "$PROJECT_DIR"

GIT_BRANCH="$(git branch --show-current)"
GIT_COMMIT="$(git rev-parse --short HEAD)"

echo "===== GPU CHECK ====="
nvidia-smi
echo "====================="

PIDS=()

for IDX in "${!ANDA_TARGETS[@]}"; do

    GPU_ID="$IDX"
    ANDA_TARGET="${ANDA_TARGETS[$IDX]}"

    TARGET_LABEL="${ANDA_TARGET/./p}"
    EPSILON_LABEL="${MOU_EPSILON/./p}"

    EXPERIMENT_NAME="target-${TARGET_LABEL}-akimg-${ANDA_KIMG}-epsilon-${EPSILON_LABEL}-seed-${SEED}"
    OUTDIR="${OUTDIR_ROOT}/${EXPERIMENT_NAME}"
    LOGFILE="${LOGDIR}/${EXPERIMENT_NAME}.log"

    mkdir -p "$OUTDIR"

    cat > "${OUTDIR}/submitted_configuration.txt" <<EOF
experiment=${EXPERIMENT_NAME}
gpu=${GPU_ID}
anda_target=${ANDA_TARGET}
anda_interval=${ANDA_INTERVAL}
anda_kimg=${ANDA_KIMG}
mou_epsilon=${MOU_EPSILON}
pseudo_weight=0.20
generated_weight=0.80
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
          --metrics=fid50k_full \
          --anda-target="$ANDA_TARGET" \
          --anda-interval="$ANDA_INTERVAL" \
          --anda-kimg="$ANDA_KIMG" \
          --mou-epsilon="$MOU_EPSILON"

    ) > "$LOGFILE" 2>&1 &

    PIDS+=("$!")
done

STATUS=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        STATUS=1
    fi
done

if [ "$STATUS" -eq 0 ]; then
    echo "All target recalibration experiments completed successfully."
else
    echo "One or more target experiments failed."
fi

echo "Stopping RunPod..."
runpodctl pod stop "$RUNPOD_POD_ID" || true

exit "$STATUS"