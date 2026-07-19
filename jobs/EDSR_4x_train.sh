#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 20:00:00
#SBATCH -J EDSR_x4
#SBATCH -o /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/edsr_x4_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/edsr_x4_%j.err

set -euo pipefail

source /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/envs/super-resolution-pytorch/bin/activate

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/submodules/EDSR-PyTorch/src

python main.py \
    --template EDSR_paper \
    --scale 4 \
    --save edsr_x4_scratch \
    --dir_data /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/dataset \
    --reset