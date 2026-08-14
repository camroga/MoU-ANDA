#!/bin/bash
#SBATCH --job-name=mou-anda-target-recalibration
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/scratch2/users/40491193/workspace/thesis/calibration/logs/%x-%A_%a.out
#SBATCH --error=/mnt/scratch2/users/40491193/workspace/thesis/calibration/logs/%x-%A_%a.err

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

source /users/40491193/venvs/anda_old/bin/activate

PROJECT_DIR="/mnt/scratch2/users/40491193/workspace/thesis/calibration/Training_ANDA"
DATASET="/users/40491193/obama.zip"
OUTDIR_ROOT="/mnt/scratch2/users/40491193/workspace/thesis/calibration/runs/mou-target-recalibration"

# Values to evaluate.
ANDA_TARGETS=(0.20 0.40 0.60)

# Fixed values.
ANDA_INTERVAL=4
ANDA_KIMG=400
MOU_EPSILON=0.20
SEED=1
TRAIN_KIMG=400

NUM_EXPERIMENTS=${#ANDA_TARGETS[@]}

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= NUM_EXPERIMENTS )); then
    echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2
    echo "Expected an index between 0 and $((NUM_EXPERIMENTS - 1))." >&2
    exit 1
fi

# Select target according to the Slurm array index.
IDX=${SLURM_ARRAY_TASK_ID}

ANDA_TARGET="${ANDA_TARGETS[$IDX]}"

TARGET_LABEL="${ANDA_TARGET/./p}"
EPSILON_LABEL="${MOU_EPSILON/./p}"

EXPERIMENT_NAME="target-${TARGET_LABEL}-akimg-${ANDA_KIMG}-epsilon-${EPSILON_LABEL}-seed-${SEED}"
OUTDIR="${OUTDIR_ROOT}/${EXPERIMENT_NAME}"

cd "$PROJECT_DIR"

mkdir -p "$OUTDIR"

GIT_BRANCH="$(git branch --show-current)"
GIT_COMMIT="$(git rev-parse --short HEAD)"

echo "=========================================="
echo "MoU-independent ANDA target recalibration"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Array job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Array task:       ${SLURM_ARRAY_TASK_ID}"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "ANDA target:      ${ANDA_TARGET}"
echo "ANDA interval:    ${ANDA_INTERVAL}"
echo "ANDA kimg:        ${ANDA_KIMG}"
echo "MoU epsilon:      ${MOU_EPSILON}"
echo "NDA mixture:      0.20 pseudo + 0.80 generated"
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
pseudo_weight=0.20
generated_weight=0.80
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