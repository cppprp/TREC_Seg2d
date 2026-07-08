#!/bin/bash
#SBATCH --job-name=trec_l40s_test
#SBATCH --nodes=1
#SBATCH --partition=gpu-el8
#SBATCH --gpus=1
#SBATCH --constraint=gpu=L40s
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=193411
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_l40s_test_%j.log

mkdir -p logs

source ~/.bashrc

# Pin the interpreter by full path — two conda envs are both named 'trec_seg'
# (mamba/Python 3.10/zarr 2 and miniforge3/Python 3.11/zarr 3); avoid the
# ambiguous `conda activate trec_seg`.
PYTHON=/home/asvetlove/miniforge3/envs/trec_seg/bin/python

export WANDB_DIR=/g/duke-mirror/asvetlove/TREC_seg/results/wandb

$PYTHON train.py \
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
    --workers       32 \
    --wandb \
    --wandb_project plankton_2d \
    --wandb_run     l40s_test
