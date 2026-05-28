#!/bin/bash
#SBATCH --job-name=trec_a40_test
#SBATCH --nodes=1
#SBATCH --partition=gpu-el8
#SBATCH --gpus=1
#SBATCH --constraint=gpu=A40
#SBATCH --cpus-per-gpu=64
#SBATCH --mem-per-gpu=514286
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_a40_test_%j.log

mkdir -p logs

source ~/.bashrc
conda activate trec_seg

export WANDB_DIR=/g/duke-mirror/asvetlove/TREC_seg/results/wandb

python train.py \
    --patches  /g/duke-mirror/asvetlove/TREC_seg/training_data/ \
    --project  /g/duke-mirror/asvetlove/TREC_seg/results_a40_test \
    --epochs         5 \
    --batch_size    64 \
    --n_channels     3 \
    --input_size   256 \
    --max_tilt      90 \
    --encoder  resnet34 \
    --pretrained \
    --loss       mcc_ce \
    --patience       0 \
    --workers       64 \
    --wandb \
    --wandb_project plankton_2d \
    --wandb_run     a40_test
