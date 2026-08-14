#!/bin/bash
#SBATCH --job-name=anda-original-baseline
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/scratch2/users/40491193/workspace/thesis/baseline/logs/%x-%A_%a.out
#SBATCH --error=/mnt/scratch2/users/40491193/workspace/thesis/baseline/logs/%x-%A_%a.err

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

source /users/40491193/venvs/anda_old/bin/activate

PROJECT_DIR="/mnt/scratch2/users/40491193/workspace/thesis/baseline"

OBAMA_DATASET="/users/40491193/obama.zip"
PANDA_DATASET="/users/40491193/100-shot-panda.zip"

OUTDIR_ROOT="/mnt/scratch2/users/40491193/workspace/thesis/baseline/runs/anda-original"

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

NUM_SEEDS=${#SEEDS[@]}
NUM_DATASETS=${#DATASETS[@]}
TOTAL_EXPERIMENTS=$((NUM_DATASETS * NUM_SEEDS))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_EXPERIMENTS )); then
    echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

DATASET="${DATASETS[$DATASET_INDEX]}"
DATASET_NAME="${DATASET_NAMES[$DATASET_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

EXPERIMENT_NAME="anda-original-${DATASET_NAME}-seed-${SEED}"
OUTDIR="${OUTDIR_ROOT}/${EXPERIMENT_NAME}"

cd "$PROJECT_DIR"

mkdir -p "$OUTDIR"

GIT_BRANCH="$(git branch --show-current)"
GIT_COMMIT="$(git rev-parse --short HEAD)"

echo "=========================================="
echo "Original ANDA baseline"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Array job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Array task:       ${SLURM_ARRAY_TASK_ID}"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "Dataset:          ${DATASET_NAME}"
echo "Dataset path:     ${DATASET}"
echo "Training kimg:    ${TRAIN_KIMG}"
echo "Seed:             ${SEED}"
echo "Output:           ${OUTDIR}"
echo "Git branch:       ${GIT_BRANCH}"
echo "Git commit:       ${GIT_COMMIT}"
echo "=========================================="

cat > "${OUTDIR}/submitted_configuration.txt" <<EOF
experiment=${EXPERIMENT_NAME}
method=original_anda
dataset=${DATASET_NAME}
dataset_path=${DATASET}
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
  --metrics=fid50k_full