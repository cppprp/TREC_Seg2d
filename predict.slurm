#!/bin/bash
#SBATCH --job-name=trec_predict
#SBATCH --nodes=1
#SBATCH --partition=gpu-el8
#SBATCH --gpus=1
#SBATCH --constraint=gpu=H100
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-gpu=128000
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --tmp=300G
#SBATCH --output=logs/predict_%j.log

# Alternative GPU constraints (uncomment to use):
# #SBATCH --constraint=gpu=A100    # 80 GB, also excellent for fp16
# #SBATCH --constraint=gpu=L40s    # 48 GB, good inference throughput
# #SBATCH --constraint=gpu=A40     # 48 GB
# #SBATCH --constraint=gpu=3090    # 24 GB, sufficient for this model

mkdir -p logs

source ~/.bashrc

# Pin the interpreter by full path: two conda envs are both named 'trec_seg'
# (a mamba one on Python 3.10 / zarr 2, and this miniforge3 one on Python 3.11 /
# zarr 3). Only the zarr-3 env writes readable v3 stores, so reference it directly.
PYTHON=/home/asvetlove/miniforge3/envs/trec_seg/bin/python

NORM_MIN=$($PYTHON -c "from config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.norm_min)")
NORM_MAX=$($PYTHON -c "from config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.norm_max)")

$PYTHON predict.py \
    --input      /scratch/asvetlove/TREC/IMATREC_volumes/POR_10to40_20231023_AM_01_epo_02 \
    --project    /g/duke-mirror/asvetlove/TREC_seg/results \
    --output     /scratch/asvetlove/TREC/predictions \
    --input_size 256 \
    --n_channels   3 \
    --overlap    0.25 \
    --axes       0 1 2 \
    --save_tiff \
    --norm_min   "$NORM_MIN" \
    --norm_max   "$NORM_MAX"
