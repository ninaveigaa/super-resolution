#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 20:00:00
#SBATCH -J EDSR_scratch_pemfc_x4
#SBATCH -o /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/EDSR_scratch_pemfc_x4_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/EDSR_scratch_pemfc_x4_%j.err

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/
source envs/super-resolution-pytorch/bin/activate

cd submodules/EDSR-PyTorch/src

python main.py \
    --model EDSR \
    --scale 4 \
    --patch_size 192 \
    --batch_size 16 \
    --epochs 300 \
    --decay 200 \
    --lr 1e-4 \
    --n_resblocks 16 \
    --n_feats 64 \
    --data_train PEMFC \
    --data_test PEMFC \
    --save edsr_scratch_pemfc_x4 \
    --save_models \
    --reset \
    --dir_data /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/data/processed
