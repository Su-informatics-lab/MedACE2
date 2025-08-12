#!/bin/bash
#SBATCH --job-name=gptoss_server_tp2
#SBATCH --partition=nextgen-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=USR1@120

set -euo pipefail

# caches & image
export HF_HOME="$HOME/hf-cache"
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"
mkdir -p "$HF_HOME" "$IMG_DIR" logs

# per-job scratch
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH"

# cuda toolchain
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# scale tp with allocation (fallback to 2)
export TENSOR_PARALLEL="${SLURM_JOB_GPUS_PER_NODE:-2}"

# hpc-friendly nccl setting (intra-node only)
export NCCL_IB_DISABLE=1

# choose a port (allow override: export PORT=8010)
export PORT="${PORT:-8000}"

echo "starting gpt-oss-120b tp=${TENSOR_PARALLEL} | job ${SLURM_JOB_ID} on $(hostname -f)"
echo "endpoint: http://$(hostname -f):${PORT}"
date
nvidia-smi || true

apptainer exec --nv --cleanenv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env PATH="$PATH" \
  --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  --env BASH_ENV=/dev/null \
  "$IMG_FILE" \
  bash -lc "
    set -euo pipefail
    PYDEPS=/tmp/pydeps
    mkdir -p \"\$PYDEPS\"
    python3 -m pip install --no-cache-dir --target \"\$PYDEPS\" 'numpy<2.3' >/tmp/pip.out 2>&1 || cat /tmp/pip.out
    export PYTHONPATH=\"\$PYDEPS:\${PYTHONPATH-}\"
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
    python3 -m vllm.entrypoints.openai.api_server \
      --model openai/gpt-oss-120b \
      --tensor-parallel-size \"$TENSOR_PARALLEL\" \
      --gpu-memory-utilization 0.90 \
      --max-model-len 131072 \
      --host 0.0.0.0 --port \"$PORT\"
  "

echo "job finished at $(date)"
