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
conda activate trec_seg

export WANDB_DIR=/g/duke-mirror/asvetlove/TREC_seg/results/wandb


python train.py \
    --patches      /g/duke-mirror/asvetlove/TREC_seg/training_data/ \
    --project      /g/duke-mirror/asvetlove/TREC_seg/results_fixed \
    --architecture U-Net \
    --encoder      resnet34 \
    --batch_size   64 \
    --workers      24 \
    --wandb \
    --wandb_project plankton_2d
