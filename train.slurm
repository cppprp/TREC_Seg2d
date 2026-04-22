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


python train.py \
    --patches  /g/duke-mirror/asvetlove/TREC_seg/training_data/ \
    --project  /g/duke-mirror/asvetlove/TREC_seg/results \
    --epochs       100 \
    --batch_size    64 \
    --n_channels     3 \
    --input_size   256 \
    --max_tilt      90 \
    --encoder  resnet34 \
    --pretrained \
    --loss       mcc_ce \
    --patience      20 \
    --workers       24 \
    --wandb \
    --wandb_project plankton_2d
