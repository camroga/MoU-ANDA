#!/bin/bash

set -euo pipefail


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------

if [ "$#" -ne 2 ]; then
    echo "Usage:"
    echo "$0 <lambda> <first_gpu>"
    echo
    echo "Examples:"
    echo "$0 0.15 0"
    echo "$0 0.25 2"
    echo "$0 0.30 4"
    exit 1
fi

LAMBDA="$1"
GPU_START="$2"
GPU_SECOND=$((GPU_START + 1))

LAMBDA_LABEL="${LAMBDA/./p}"


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

export PATH="/opt/conda/bin:$PATH"

SCRIPT_DIR=$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)

PROJECT_DIR=$(
    cd "${SCRIPT_DIR}/.."
    pwd
)

PERSISTENT_DATASET=\
"/workspace/thesis/100-shot-obama.zip"

PERSISTENT_OUTDIR=\
"/workspace/thesis/calibration/runs/mou-independent-anda-phase3"

PERSISTENT_LOGDIR=\
"/workspace/thesis/calibration/logs-phase3"

LOCAL_ROOT=\
"/tmp/mou-anda-phase3-lambda-${LAMBDA_LABEL}"

DATASET=\
"${LOCAL_ROOT}/100-shot-obama.zip"

OUTDIR_ROOT=\
"${LOCAL_ROOT}/runs"

LOGDIR=\
"${LOCAL_ROOT}/logs"


# ---------------------------------------------------------
# PHASE 3 CONFIGURATION
# ---------------------------------------------------------

ANDA_TARGET=0.70
ANDA_INTERVAL=4

SEED=1
TRAIN_KIMG=400

# Two configurations selected from Phase 2.
CONFIGURATIONS=(
    "750:0.25"
    "500:0.25"
)


# ---------------------------------------------------------
# PREPARE STORAGE
# ---------------------------------------------------------

mkdir -p "$PERSISTENT_OUTDIR"
mkdir -p "$PERSISTENT_LOGDIR"

rm -rf "$LOCAL_ROOT"

mkdir -p "$OUTDIR_ROOT"
mkdir -p "$LOGDIR"

cp "$PERSISTENT_DATASET" "$DATASET"


# ---------------------------------------------------------
# START
# ---------------------------------------------------------

cd "$PROJECT_DIR"

echo "=========================================="
echo "Phase 3 - Lambda calibration"
echo "=========================================="
echo "Lambda:         ${LAMBDA}"
echo "GPUs:           ${GPU_START}, ${GPU_SECOND}"
echo "Target:         ${ANDA_TARGET}"
echo "Training kimg:  ${TRAIN_KIMG}"
echo "Local root:     ${LOCAL_ROOT}"
echo "=========================================="
echo

df -h /tmp
echo

PIDS=()


# ---------------------------------------------------------
# RUN TWO EXPERIMENTS
# ---------------------------------------------------------

for INDEX in "${!CONFIGURATIONS[@]}"; do

    CONFIG="${CONFIGURATIONS[$INDEX]}"

    ANDA_KIMG="${CONFIG%%:*}"
    MOU_EPSILON="${CONFIG##*:}"

    GPU_ID=$((GPU_START + INDEX))

    TARGET_LABEL="${ANDA_TARGET/./p}"
    EPSILON_LABEL="${MOU_EPSILON/./p}"

    EXPERIMENT_NAME=\
"target-${TARGET_LABEL}-akimg-${ANDA_KIMG}-epsilon-${EPSILON_LABEL}-lambda-${LAMBDA_LABEL}-seed-${SEED}"

    OUTDIR=\
"${OUTDIR_ROOT}/${EXPERIMENT_NAME}"

    LOGFILE=\
"${LOGDIR}/${EXPERIMENT_NAME}.log"

    mkdir -p "$OUTDIR"


    # -----------------------------------------------------
    # SAVE CONFIGURATION
    # -----------------------------------------------------

    cat > \
"${OUTDIR}/submitted_configuration.txt" <<EOF
experiment=${EXPERIMENT_NAME}
anda_target=${ANDA_TARGET}
anda_interval=${ANDA_INTERVAL}
anda_kimg=${ANDA_KIMG}
mou_epsilon=${MOU_EPSILON}
lambda=${LAMBDA}
training_kimg=${TRAIN_KIMG}
seed=${SEED}
gpu=${GPU_ID}
controller_direction=real_anda_minus_target
EOF


    # -----------------------------------------------------
    # PRINT CONFIGURATION
    # -----------------------------------------------------

    echo "Experiment:  ${EXPERIMENT_NAME}"
    echo "GPU:         ${GPU_ID}"
    echo "K ANDA:      ${ANDA_KIMG}"
    echo "Epsilon:     ${MOU_EPSILON}"
    echo "Lambda:      ${LAMBDA}"
    echo


    # -----------------------------------------------------
    # START TRAINING
    # -----------------------------------------------------

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
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
        --mou-epsilon="$MOU_EPSILON" \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)

done


# ---------------------------------------------------------
# WAIT
# ---------------------------------------------------------

echo "Waiting for lambda ${LAMBDA} experiments..."

FAILED=0

for PID in "${PIDS[@]}"; do

    if ! wait "$PID"; then
        FAILED=1
    fi

done


# ---------------------------------------------------------
# COPY RESULTS TO PERSISTENT STORAGE
# ---------------------------------------------------------

echo
echo "Copying results to persistent storage..."

cp -a \
    "${OUTDIR_ROOT}/." \
    "${PERSISTENT_OUTDIR}/"

cp -a \
    "${LOGDIR}/." \
    "${PERSISTENT_LOGDIR}/"


# ---------------------------------------------------------
# CHECK STATUS
# ---------------------------------------------------------

if (( FAILED != 0 )); then

    echo
    echo "ERROR:"
    echo "At least one experiment failed."
    echo
    echo "Local files were preserved at:"
    echo "${LOCAL_ROOT}"

    exit 1

fi


# ---------------------------------------------------------
# SUCCESS
# ---------------------------------------------------------

echo
echo "=========================================="
echo "Lambda ${LAMBDA} completed successfully."
echo "=========================================="

echo
echo "Results:"
echo "${PERSISTENT_OUTDIR}"

echo
echo "Logs:"
echo "${PERSISTENT_LOGDIR}"

echo
echo "Cleaning local files..."

rm -rf "$LOCAL_ROOT"

echo "Done."