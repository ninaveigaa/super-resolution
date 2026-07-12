#!/usr/bin/env bash
# jobs/Wang_2023_sanity.sh

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1 virtualenv/20.23.1-GCCcore-12.3.0
source super-resolution/bin/activate

ARGS=(
    --phase train
    --epoch 1
    --itersPerEpoch 2
    --iterCyclesPerEpoch 1
    --print_freq 1
    --valNum 1
    --fine_size 16
    --dataset_dir "data/processed/Wang_2023/training/"
    --modelName Wang_2023_sanity
)

python pipelines/Wang_2023_runDualSRNetSlimCoupled.py "${ARGS[@]}"
