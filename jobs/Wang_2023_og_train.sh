#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 20:00:00
#SBATCH -J Wang_2023_og_train
#SBATCH -o /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/Wang_2023_og_train_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/Wang_2023_og_train_%j.err
module purge
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
cd /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/
source envs/super-resolution-tensorflow-og/bin/activate
python Dual-EDSR_OG/runDualSRNetSlimCoupled.py \
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
    --dataset_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/data/processed/Wang_2023/training_og/ \
    --checkpoint_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/checkpoints
