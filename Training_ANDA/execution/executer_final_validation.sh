#!/bin/bash
set -u

export PATH="/opt/conda/bin:$PATH"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.."; pwd)

LOCAL_ROOT="/tmp/mou-anda-final-validation"
LOCAL_DATA="${LOCAL_ROOT}/data"
LOCAL_OUTDIR="${LOCAL_ROOT}/runs"
LOCAL_LOGDIR="${LOCAL_ROOT}/logs"

PERSISTENT_OUTDIR="/workspace/thesis/calibration/runs/mou-independent-anda-final-validation"
PERSISTENT_LOGDIR="/workspace/thesis/calibration/logs-final-validation"

OBAMA_SRC="/workspace/thesis/100-shot-obama.zip"
PANDA_SRC="/workspace/thesis/100-shot-panda.zip"

ANDA_TARGET=0.70
ANDA_INTERVAL=4
ANDA_KIMG=750
MOU_EPSILON=0.25
TRAIN_KIMG=500

rm -rf "${LOCAL_ROOT}"
mkdir -p "${LOCAL_DATA}" "${LOCAL_OUTDIR}" "${LOCAL_LOGDIR}"
mkdir -p "${PERSISTENT_OUTDIR}" "${PERSISTENT_LOGDIR}"

cp "${OBAMA_SRC}" "${LOCAL_DATA}/100-shot-obama.zip"
cp "${PANDA_SRC}" "${LOCAL_DATA}/100-shot-panda.zip"

cd "${PROJECT_DIR}"

PIDS=()

launch_run () {
    DATASET="$1"
    SEED="$2"
    GPU="$3"

    DATA_FILE="${LOCAL_DATA}/100-shot-${DATASET}.zip"

    EXPERIMENT_NAME="${DATASET}-target-0p70-akimg-750-epsilon-0p25-lambda-0p20-seed-${SEED}"

    OUTDIR="${LOCAL_OUTDIR}/${EXPERIMENT_NAME}"
    LOGFILE="${LOCAL_LOGDIR}/${EXPERIMENT_NAME}.log"

    mkdir -p "${OUTDIR}"

    echo "Launching ${EXPERIMENT_NAME} on GPU ${GPU}"

    CUDA_VISIBLE_DEVICES="${GPU}" python -u train.py \
        --outdir="${OUTDIR}" \
        --data="${DATA_FILE}" \
        --cfg=low_shot \
        --mirror=true \
        --gpus=1 \
        --seed="${SEED}" \
        --kimg="${TRAIN_KIMG}" \
        --snap=10 \
        --metrics=fid50k_full \
        --anda-target="${ANDA_TARGET}" \
        --anda-interval="${ANDA_INTERVAL}" \
        --anda-kimg="${ANDA_KIMG}" \
        --mou-epsilon="${MOU_EPSILON}" \
        > "${LOGFILE}" 2>&1 &

    PIDS+=($!)
}

launch_run obama 2 0
launch_run obama 3 1
launch_run obama 4 2

launch_run panda 2 3
launch_run panda 3 4
launch_run panda 4 5


# Remove snapshots only after their FID evaluation has been
# written to metric-fid50k_full.jsonl.
# Keep the final 500-kimg snapshot.
cleanup_snapshots () {
    while true; do
        sleep 60

        find "${LOCAL_OUTDIR}" \
            -name "metric-fid50k_full.jsonl" \
            -type f 2>/dev/null |
        while read -r METRIC_FILE; do

            RUN_DIR=$(dirname "${METRIC_FILE}")

            for PKL in "${RUN_DIR}"/network-snapshot-*.pkl; do
                [ -e "${PKL}" ] || continue

                BASE=$(basename "${PKL}")

                if [ "${BASE}" = "network-snapshot-000500.pkl" ]; then
                    continue
                fi

                if grep -q "\"snapshot_pkl\": \"${BASE}\"" "${METRIC_FILE}"; then
                    rm -f "${PKL}"
                fi
            done
        done
    done
}

cleanup_snapshots &
CLEANUP_PID=$!

FAILED=0

for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAILED=1
    fi
done

kill "${CLEANUP_PID}" 2>/dev/null || true
wait "${CLEANUP_PID}" 2>/dev/null || true

echo "Copying results to persistent storage..."

cp -a "${LOCAL_OUTDIR}/." "${PERSISTENT_OUTDIR}/"
cp -a "${LOCAL_LOGDIR}/." "${PERSISTENT_LOGDIR}/"

if [ "${FAILED}" -ne 0 ]; then
    echo "One or more experiments failed."
    echo "Local files preserved at ${LOCAL_ROOT}"
    exit 1
fi

echo "All final-validation experiments completed successfully."

rm -rf "${LOCAL_ROOT}"