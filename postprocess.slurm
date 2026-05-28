#!/bin/bash
#SBATCH --job-name=trec_postprocess
#SBATCH --nodes=1
#SBATCH --partition=gpu-training
#SBATCH --gpus=1
#SBATCH --constraint=gpu=H200
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-gpu=257660
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/postprocess_%j.log

mkdir -p logs

source ~/.bashrc
conda activate trec_seg


INPUT=/scratch/asvetlove/project_2d/predictions/volume.tif
OUTPUT=/scratch/asvetlove/project_2d/instances/volume.tif

python postprocess.py \
    --input        $INPUT \
    --output       $OUTPUT \
    --fg_threshold 0.5 \
    --bd_threshold 0.4 \
    --min_size     500
