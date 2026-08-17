#!/bin/bash

set -euo pipefail


# ---------------------------------------------------------
# RUNPOD ENVIRONMENT
# ---------------------------------------------------------

export PATH="/opt/conda/bin:$PATH"

PROJECT_DIR="/workspace/thesis/calibration/Training_ANDA"

DATASET="/workspace/thesis/100-shot-obama.zip"

OUTDIR_ROOT=\
"/workspace/thesis/calibration/runs/mou-independent-anda-grid"

LOGDIR=\
"/workspace/thesis/calibration/logs-phase2"

mkdir -p "$OUTDIR_ROOT"
mkdir -p "$LOGDIR"


# ---------------------------------------------------------
# PHASE 2 CALIBRATION GRID
# ---------------------------------------------------------

# Best target selected during Phase 1.
ANDA_TARGET=0.70

# Values to calibrate.
ANDA_KIMGS=(
    400
    500
    750
)

MOU_EPSILONS=(
    0.15
    0.20
    0.25
    0.30
)

# Fixed values.
ANDA_INTERVAL=4
SEED=1
TRAIN_KIMG=400

# RunPod configuration.
NUM_GPUS=6


# ---------------------------------------------------------
# INITIAL STATUS
# ---------------------------------------------------------

STATUS=0


# ---------------------------------------------------------
# CHECK ENVIRONMENT
# ---------------------------------------------------------

cd "$PROJECT_DIR"

echo "=========================================="
echo "Phase 2 MoU-ANDA calibration"
echo "=========================================="
echo "Project:        ${PROJECT_DIR}"
echo "Dataset:        ${DATASET}"
echo "ANDA target:    ${ANDA_TARGET}"
echo "ANDA interval:  ${ANDA_INTERVAL}"
echo "Training kimg:  ${TRAIN_KIMG}"
echo "Seed:           ${SEED}"
echo "GPUs:           ${NUM_GPUS}"
echo "Output root:    ${OUTDIR_ROOT}"
echo "Logs:           ${LOGDIR}"
echo "=========================================="

echo
echo "===== GPU CHECK ====="

which python

python - <<'EOF'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

for gpu in range(torch.cuda.device_count()):
    print(
        f"GPU {gpu}:",
        torch.cuda.get_device_name(gpu)
    )
EOF

nvidia-smi

echo "====================="
echo


# ---------------------------------------------------------
# CHECK NUMBER OF GPUS
# ---------------------------------------------------------

AVAILABLE_GPUS=$(
    nvidia-smi -L | wc -l
)

if (( AVAILABLE_GPUS < NUM_GPUS )); then

    echo "ERROR:"
    echo "Expected at least ${NUM_GPUS} GPUs,"
    echo "but only ${AVAILABLE_GPUS} were detected."

    exit 1

fi


# ---------------------------------------------------------
# GIT INFORMATION
# ---------------------------------------------------------

GIT_BRANCH=$(
    git rev-parse --abbrev-ref HEAD
)

GIT_COMMIT=$(
    git rev-parse --short HEAD
)

echo "Git branch: ${GIT_BRANCH}"
echo "Git commit: ${GIT_COMMIT}"
echo


# ---------------------------------------------------------
# CREATE EXPERIMENT LIST
# ---------------------------------------------------------

EXPERIMENTS=()

for ANDA_KIMG in "${ANDA_KIMGS[@]}"; do

    for MOU_EPSILON in "${MOU_EPSILONS[@]}"; do

        EXPERIMENTS+=(
            "${ANDA_KIMG}:${MOU_EPSILON}"
        )

    done

done

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

echo "Total experiments: ${TOTAL_EXPERIMENTS}"
echo


# ---------------------------------------------------------
# RUN EXPERIMENTS
# ---------------------------------------------------------

for ((
    START=0;
    START<TOTAL_EXPERIMENTS;
    START+=NUM_GPUS
)); do

    END=$((START + NUM_GPUS))

    if (( END > TOTAL_EXPERIMENTS )); then
        END=$TOTAL_EXPERIMENTS
    fi

    BATCH=$((START / NUM_GPUS + 1))

    echo "=========================================="
    echo "Starting batch ${BATCH}"
    echo "Experiments ${START} to $((END - 1))"
    echo "=========================================="

    PIDS=()

    for ((
        INDEX=START;
        INDEX<END;
        INDEX++
    )); do

        GPU_ID=$((INDEX - START))

        CONFIG="${EXPERIMENTS[$INDEX]}"

        ANDA_KIMG="${CONFIG%%:*}"
        MOU_EPSILON="${CONFIG##*:}"

        TARGET_LABEL="${ANDA_TARGET/./p}"
        EPSILON_LABEL="${MOU_EPSILON/./p}"

        EXPERIMENT_NAME=\
"target-${TARGET_LABEL}-akimg-${ANDA_KIMG}-epsilon-${EPSILON_LABEL}-seed-${SEED}"

        OUTDIR=\
"${OUTDIR_ROOT}/${EXPERIMENT_NAME}"

        LOGFILE=\
"${LOGDIR}/${EXPERIMENT_NAME}.log"

        mkdir -p "$OUTDIR"


        # -------------------------------------------------
        # SAVE CONFIGURATION
        # -------------------------------------------------

        cat > \
"${OUTDIR}/submitted_configuration.txt" <<EOF
experiment=${EXPERIMENT_NAME}
anda_target=${ANDA_TARGET}
anda_interval=${ANDA_INTERVAL}
anda_kimg=${ANDA_KIMG}
mou_epsilon=${MOU_EPSILON}
pseudo_weight=0.2
generated_weight=0.8
training_kimg=${TRAIN_KIMG}
seed=${SEED}
gpu=${GPU_ID}
git_branch=${GIT_BRANCH}
git_commit=${GIT_COMMIT}
controller_direction=real_anda_minus_target
EOF


        # -------------------------------------------------
        # PRINT CONFIGURATION
        # -------------------------------------------------

        echo
        echo "Experiment:   ${EXPERIMENT_NAME}"
        echo "GPU:          ${GPU_ID}"
        echo "ANDA target:  ${ANDA_TARGET}"
        echo "ANDA kimg:    ${ANDA_KIMG}"
        echo "MoU epsilon:  ${MOU_EPSILON}"
        echo "Output:       ${OUTDIR}"
        echo "Log:          ${LOGFILE}"


        # -------------------------------------------------
        # START TRAINING
        # -------------------------------------------------

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


    # -----------------------------------------------------
    # WAIT FOR CURRENT BATCH
    # -----------------------------------------------------

    echo
    echo "Waiting for batch ${BATCH} to finish..."

    FAILED=0

    for PID in "${PIDS[@]}"; do

        if ! wait "$PID"; then
            FAILED=1
        fi

    done

    if (( FAILED != 0 )); then

        echo
        echo "ERROR:"
        echo "At least one experiment in batch ${BATCH}"
        echo "failed. Check:"
        echo "${LOGDIR}"

        STATUS=1
        break

    else

        echo
        echo "Batch ${BATCH} completed successfully."
        echo

    fi

done


# ---------------------------------------------------------
# PHASE 2 TRAINING COMPLETE
# ---------------------------------------------------------

echo
echo "=========================================="
echo "Phase 2 training finished."
echo "=========================================="

echo
echo "Results:"
echo "${OUTDIR_ROOT}"

echo
echo "Logs:"
echo "${LOGDIR}"
echo


# ---------------------------------------------------------
# RUN PHASE 2 ANALYSIS
# ---------------------------------------------------------

if [ "$STATUS" -eq 0 ]; then

    echo "Running Phase 2 analysis..."
    echo

    if python analyse_phase2.py; then

        echo
        echo "=========================================="
        echo "Phase 2 analysis completed."
        echo "=========================================="

    else

        echo
        echo "Phase 2 analysis failed."

        STATUS=1

    fi

fi


# ---------------------------------------------------------
# STOP RUNPOD
# ---------------------------------------------------------

if [ "$STATUS" -eq 0 ]; then

    echo
    echo "All Phase 2 experiments completed successfully."

    if [ -n "${RUNPOD_POD_ID:-}" ]; then

        echo "Stopping RunPod..."

        runpodctl pod stop \
            "$RUNPOD_POD_ID" || true

    else

        echo \
"RUNPOD_POD_ID not set; stop the Pod manually."

    fi

else

    echo
    echo "One or more Phase 2 experiments failed."
    echo "RunPod will remain running for debugging."

fi

exit "$STATUS"