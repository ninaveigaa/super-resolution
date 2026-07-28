#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 20:00:00
#SBATCH -J Wang_2023_train_og_wandb
#SBATCH -o /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/Wang_2023_train_og_wandb%j.out
#SBATCH -e /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/Wang_2023_train_og_wandb%j.err

module purge
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/
source envs/super-resolution-tensorflow-og/bin/activate

export WANDB_API_KEY=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
export WANDB_MODE=offline
export WANDB_DIR=/mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/wandb
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/:$PYTHONPATH

python pipelines/Wang_2023_runDualSRNetSlimCoupled_wandb.py \
    --epoch 500 \
    --itersPerEpoch 300 \
    --iterCyclesPerEpoch 3 \
    --batch_size 32 \
    --fine_size 48 \
    --lr 1e-4 \
    --epoch_step 150 \
    --scale 4 \
    --ngsrf 64 \
    --numResBlocks 16 \
    --ganFlag False \
    --save_freq 10 \
    --print_freq 10 \
    --valNum 225 \
    --gpuIDs 0 \
    --modelName Wang_2023_og \
    --dataset_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/data/processed/Wang_2023_og/training/ \
    --checkpoint_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/checkpoints
