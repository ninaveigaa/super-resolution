#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 20:00:00
#SBATCH -J HAT_finetune
#SBATCH -o /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/hat_finetune_wang2023_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/hat_finetune_wang2023_%j.err

set -euo pipefail

source /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/super-resolution-pytorch/bin/activate

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/submodules/HAT

python hat/train.py -opt /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/submodules/HAT/options/train/train_HAT_SRx4_finetune_Wang2023.yml
