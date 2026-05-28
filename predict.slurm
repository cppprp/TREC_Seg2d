#!/bin/bash
#SBATCH --job-name=trec_predict
#SBATCH --nodes=1
#SBATCH --partition=gpu-el8
#SBATCH --gpus=1
#SBATCH --constraint=gpu=H100
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-gpu=60000
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/predict_%j.log

# Alternative GPU constraints (uncomment to use):
# #SBATCH --constraint=gpu=A100    # 80 GB, also excellent for fp16
# #SBATCH --constraint=gpu=L40s    # 48 GB, good inference throughput
# #SBATCH --constraint=gpu=A40     # 48 GB
# #SBATCH --constraint=gpu=3090    # 24 GB, sufficient for this model

mkdir -p logs

source ~/.bashrc
conda activate trec_seg

python predict.py \
    --input      /scratch/asvetlove/volumes/*.tif \
    --project    /scratch/asvetlove/project_2d \
    --input_size 256 \
    --n_channels   3 \
    --overlap    0.25 \
    --axes       0 1 2
