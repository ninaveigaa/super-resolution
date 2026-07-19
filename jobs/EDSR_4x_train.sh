#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 20:00:00
#SBATCH -J EDSR_x4
#SBATCH -o logs/edsr_x4_%j.out
#SBATCH -e logs/edsr_x4_%j.err

set -euo pipefail

source envs/super-resolution-pytorch/bin/activate

cd submodules/EDSR-PyTorch/src

python main.py \
    --template EDSR_paper \
    --scale 4 \
    --save edsr_x4_scratch \
    --dir_data ../../../data/raw \
    --reset