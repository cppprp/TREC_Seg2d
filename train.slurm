#!/bin/bash
#SBATCH --job-name=trec_seg2d
#SBATCH --nodes=1
#SBATCH --partition=gpu-training
#SBATCH --gpus=1
#SBATCH --constraint=gpu=H200
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-gpu=257660
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.log

mkdir -p logs

source ~/.bashrc

# Pin the interpreter by full path: two conda envs are both named 'trec_seg'
# (a mamba one on Python 3.10 / zarr 2, and this miniforge3 one on Python 3.11 /
# zarr 3). `conda activate trec_seg` is ambiguous, so reference the env directly.
PYTHON=/home/asvetlove/miniforge3/envs/trec_seg/bin/python

export WANDB_DIR=/g/duke-mirror/asvetlove/TREC_seg/results/wandb


$PYTHON train.py \
    --patches      /g/duke-mirror/asvetlove/TREC_seg/training_data/ \
    --project      /g/duke-mirror/asvetlove/TREC_seg/results_fixed \
    --architecture U-Net \
    --encoder      resnet34 \
    --batch_size   64 \
    --workers      24 \
    --wandb \
    --wandb_project plankton_2d
