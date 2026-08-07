#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 20:00:00
#SBATCH -J DualEDSR_train_pytorch
#SBATCH -o /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/DualEDSR_train_pytorch_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/DualEDSR_train_pytorch_%j.err

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/
source envs/super-resolution-pytorch/bin/activate

python pipelines/train_pytorch.py \
    --epoch 200 \
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
    --modelName DualEDSR \
    --dataset_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/data/processed/Wang_2023/training/ \
    --val_dataset_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/data/processed/Wang_2023/validation/ \
    --checkpoint_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/checkpoints
