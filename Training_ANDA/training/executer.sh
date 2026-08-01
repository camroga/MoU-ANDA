#!/bin/bash
#SBATCH --job-name=mou-anda-grid-calibration
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-59%6
#SBATCH --output=/mnt/scratch2/users/40491193/workspace/thesis/calibration/logs/%x-%A_%a.out
#SBATCH --error=/mnt/scratch2/users/40491193/workspace/thesis/calibration/logs/%x-%A_%a.err

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

source /users/40491193/venvs/anda_old/bin/activate

PROJECT_DIR="/mnt/scratch2/users/40491193/workspace/thesis/calibration/Training_ANDA"
DATASET="/users/40491193/obama.zip"
OUTDIR_ROOT="/mnt/scratch2/users/40491193/workspace/thesis/calibration/runs/mou-independent-anda-grid"

# Values to calibrate.
ANDA_TARGETS=(0.50 0.55 0.60 0.65 0.70)
ANDA_KIMGS=(250 500 750 1000)
MOU_EPSILONS=(0.10 0.20 0.30)

# Fixed values.
ANDA_INTERVAL=4
SEED=1
TRAIN_KIMG=400

NUM_TARGETS=${#ANDA_TARGETS[@]}
NUM_KIMGS=${#ANDA_KIMGS[@]}
NUM_EPSILONS=${#MOU_EPSILONS[@]}

TOTAL_EXPERIMENTS=$((NUM_TARGETS * NUM_KIMGS * NUM_EPSILONS))


if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_EXPERIMENTS )); then
    echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2
    echo "Expected an index between 0 and $((TOTAL_EXPERIMENTS - 1))." >&2
    exit 1
fi

# Map the one-dimensional Slurm index to:
# target index, anda_kimg index, epsilon index.
TARGET_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_KIMGS * NUM_EPSILONS)))

REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_KIMGS * NUM_EPSILONS)))

KIMG_INDEX=$((REMAINDER / NUM_EPSILONS))

EPSILON_INDEX=$((REMAINDER % NUM_EPSILONS))


ANDA_TARGET="${ANDA_TARGETS[$TARGET_INDEX]}"
ANDA_KIMG="${ANDA_KIMGS[$KIMG_INDEX]}"
MOU_EPSILON="${MOU_EPSILONS[$EPSILON_INDEX]}"

TARGET_LABEL="${ANDA_TARGET/./p}"
EPSILON_LABEL="${MOU_EPSILON/./p}"

EXPERIMENT_NAME="target-${TARGET_LABEL}-akimg-${ANDA_KIMG}-epsilon-${EPSILON_LABEL}-seed-${SEED}"
OUTDIR="${OUTDIR_ROOT}/${EXPERIMENT_NAME}"


cd "$PROJECT_DIR"

mkdir -p "$OUTDIR"

GIT_BRANCH="$(git branch --show-current)"
GIT_COMMIT="$(git rev-parse --short HEAD)"

echo "=========================================="
echo "MoU-independent ANDA calibration"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Array job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Array task:       ${SLURM_ARRAY_TASK_ID}"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "ANDA target:      ${ANDA_TARGET}"
echo "ANDA interval:    ${ANDA_INTERVAL}"
echo "ANDA kimg:        ${ANDA_KIMG}"
echo "MoU epsilon:      ${MOU_EPSILON}"
echo "NDA mixture:      0.2 pseudo + 0.8 generated"
echo "Training kimg:    ${TRAIN_KIMG}"
echo "Seed:             ${SEED}"
echo "Output:           ${OUTDIR}"
echo "Git branch:       ${GIT_BRANCH}"
echo "Git commit:       ${GIT_COMMIT}"
echo "=========================================="

cat > "${OUTDIR}/submitted_configuration.txt" <<EOF
experiment=${EXPERIMENT_NAME}
anda_target=${ANDA_TARGET}
anda_interval=${ANDA_INTERVAL}
anda_kimg=${ANDA_KIMG}
mou_epsilon=${MOU_EPSILON}
pseudo_weight=0.2
generated_weight=0.8
training_kimg=${TRAIN_KIMG}
seed=${SEED}
git_branch=${GIT_BRANCH}
git_commit=${GIT_COMMIT}
slurm_job_id=${SLURM_JOB_ID}
slurm_array_job_id=${SLURM_ARRAY_JOB_ID}
slurm_array_task_id=${SLURM_ARRAY_TASK_ID}
EOF

echo "===== GPU CHECK ====="
which python

python - <<'EOF'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))
EOF

nvidia-smi
echo "====================="

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