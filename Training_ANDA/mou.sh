#!/bin/bash
#SBATCH --job-name=anda-obama-mou
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/scratch2/users/40491193/workspace/01mou/logs/%x-%j.out
#SBATCH --error=/mnt/scratch2/users/40491193/workspace/01mou/logs/%x-%j.err

set -e

module purge
module load python3/3.10.5/gcc-9.3.0

source /users/40491193/venvs/anda_old/bin/activate

PROJECT_DIR="/mnt/scratch2/users/40491193/workspace/01mou/Training_ANDA"
DATASET="/users/40491193/obama.zip"
OUTDIR="/mnt/scratch2/users/40491193/workspace/01mou/training-runs01"

cd "$PROJECT_DIR"
mkdir -p logs

echo "===== GPU CHECK ====="
which python
python - <<'EOF'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: NO GPU")
EOF
nvidia-smi
echo "====================="

python train.py \
  --outdir="$OUTDIR" \
  --data="$DATASET" \
  --mirror=true \
  --gpus=1