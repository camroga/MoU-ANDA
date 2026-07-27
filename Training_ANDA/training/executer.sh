#!/bin/bash
#SBATCH --job-name=mou-anda-calibration
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
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
OUTDIR_ROOT="/mnt/scratch2/users/40491193/workspace/thesis/calibration/runs/mou-independent-anda"

# Values to calibrate.
ANDA_TARGETS=(0.2 0.4 0.6)
ANDA_TARGET="${ANDA_TARGETS[$SLURM_ARRAY_TASK_ID]}"

# Replace the decimal point so the folder name is easier to read.
TARGET_LABEL="${ANDA_TARGET/./p}"
OUTDIR="${OUTDIR_ROOT}/target-${TARGET_LABEL}"

cd "$PROJECT_DIR"

mkdir -p "$OUTDIR"

echo "=========================================="
echo "MoU-independent ANDA calibration"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Array task:      ${SLURM_ARRAY_TASK_ID}"
echo "ANDA target:     ${ANDA_TARGET}"
echo "ANDA interval:   4"
echo "ANDA kimg:       500"
echo "MoU epsilon:     0.2"
echo "Seed:            1"
echo "Output:          ${OUTDIR}"
echo "Git branch:      $(git branch --show-current)"
echo "Git commit:      $(git rev-parse --short HEAD)"
echo "=========================================="

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
  --seed=1 \
  --kimg=80 \
  --snap=10 \
  --metrics=fid50k_full \
  --anda-target="$ANDA_TARGET" \
  --anda-interval=4 \
  --anda-kimg=500 \
  --mou-epsilon=0.2