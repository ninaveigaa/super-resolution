#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 04:00:00
#SBATCH -J Wang_2023_inference
#SBATCH -o /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/Wang_2023_inference_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/logs/Wang_2023_inference_%j.err
module purge
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
cd /mimer/NOBACKUP/groups/kthmech/nvlmds/super-resolution/
source envs/super-resolution-tensorflow/bin/activate
python scripts/Wang_2023_runDualSRNetSlimCoupled.py \
    --phase testSmall \
    --continue_train True \
    --continueEpoch 490 \
    --scale 4 \
    --ngsrf 64 \
    --numResBlocks 16 \
    --ganFlag False \
    --gpuIDs 0 \
    --modelName Wang_2023 \
    --checkpoint_dir checkpoints \
    --test_dir data/processed/Wang_2023/test/


