#!/bin/bash
#SBATCH --job-name=mou-anda-grid-mixture_calibration
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/mnt/scratch2/users/40491193/workspace/thesis/mixture_calibration/logs/%x-%A_%a.out
#SBATCH --error=/mnt/scratch2/users/40491193/workspace/thesis/mixture_calibration/logs/%x-%A_%a.err

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

source /users/40491193/venvs/anda_old/bin/activate

PROJECT_DIR="/mnt/scratch2/users/40491193/workspace/thesis/mixture_calibration/Training_ANDA"
DATASET="/users/40491193/obama.zip"
OUTDIR_ROOT="/mnt/scratch2/users/40491193/workspace/thesis/mixture_calibration/runs/mou-independent-anda-grid-mixture-lambda-0p15"

# Fixed list of the 3 selected MoU configurations.
ANDA_TARGETS=(0.55 0.50 0.70)
ANDA_KIMGS=(750 750 500)
MOU_EPSILONS=(0.30 0.20 0.10)

# Fixed values.
ANDA_INTERVAL=4
SEED=1
TRAIN_KIMG=400

# This batch corresponds to ONE manually configured lambda value.
PSEUDO_WEIGHT=0.15
GENERATED_WEIGHT=0.85
LAMBDA_LABEL="0p15"

NUM_EXPERIMENTS=${#ANDA_TARGETS[@]}

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= NUM_EXPERIMENTS )); then
    echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2
    echo "Expected an index between 0 and $((NUM_EXPERIMENTS - 1))." >&2
    exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID}

ANDA_TARGET="${ANDA_TARGETS[$IDX]}"
ANDA_KIMG="${ANDA_KIMGS[$IDX]}"
MOU_EPSILON="${MOU_EPSILONS[$IDX]}"

TARGET_LABEL="${ANDA_TARGET/./p}"
EPSILON_LABEL="${MOU_EPSILON/./p}"

EXPERIMENT_NAME="lambda-${LAMBDA_LABEL}-target-${TARGET_LABEL}-akimg-${ANDA_KIMG}-epsilon-${EPSILON_LABEL}-seed-${SEED}"
OUTDIR="${OUTDIR_ROOT}/${EXPERIMENT_NAME}"

cd "$PROJECT_DIR"

mkdir -p "$OUTDIR"

GIT_BRANCH="$(git branch --show-current)"
GIT_COMMIT="$(git rev-parse --short HEAD)"

echo "=========================================="
echo "MoU-independent ANDA mixture calibration"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Array job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Array task:       ${SLURM_ARRAY_TASK_ID}"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "ANDA target:      ${ANDA_TARGET}"
echo "ANDA interval:    ${ANDA_INTERVAL}"
echo "ANDA kimg:        ${ANDA_KIMG}"
echo "MoU epsilon:      ${MOU_EPSILON}"
echo "NDA mixture:      ${PSEUDO_WEIGHT} pseudo + ${GENERATED_WEIGHT} generated"
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
pseudo_weight=${PSEUDO_WEIGHT}
generated_weight=${GENERATED_WEIGHT}
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