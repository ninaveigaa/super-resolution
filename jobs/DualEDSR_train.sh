#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 03:00:00
#SBATCH -J DualEDSR_train
#SBATCH -o /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/DualEDSR_train_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/DualEDSR_train_%j.err

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 virtualenv/20.23.1-GCCcore-12.3.0

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/
source super-resolution-pytorch/bin/activate

mkdir -p logs metrics checkpoints

python pipelines/train_dualedsr.py \
    --epochs 1 \
    --iters_per_epoch 1 \
    --iter_cycles 1 \
    --batch_size 1 \
    --bc_depth 16 \
    --crop_size 48 \
    --lr 1e-4 \
    --epoch_step 150 \
    --scale 4 \
    --model_module edsr \
    --model_class DualEDSR \
    --model_kwargs '{"n_feats": 64, "n_resblocks": 16}' \
    --save_freq 10 \
    --val_num 20 \
    --model_name DualEDSR \
    --low_res lr.npy \
    --high_res hr.npy \
    --dataset_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/data/processed/Wang_2023/training/ \
    --val_dataset_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/data/processed/Wang_2023/validation/ \
    --checkpoint_dir /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/checkpoints