#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 12:00:00
#SBATCH -J Wang_2023_train
#SBATCH -o /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/Wang_2023_train_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/Wang_2023_train_%j.err

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1 virtualenv/20.23.1-GCCcore-12.3.0
source super-resolution/bin/activate

ARGS=(
    --phase train
    --fine_size 48
    --dataset_dir "data/processed/Wang_2023/training/"
    --modelName Wang_2023
)

python pipelines/Wang_2023_runDualSRNetSlimCoupled.py "${ARGS[@]}"
