#!/bin/bash
#SBATCH --job-name=trec_postprocess
#SBATCH --nodes=1
#SBATCH --partition=htc-el8
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/postprocess_%j.log

# CPU node (htc-el8) — has the RAM for a full-volume watershed and outbound
# internet for the WebKnossos upload. Watershed runs on skimage (CPU); for a
# GPU-accelerated watershed use a gpu partition with cucim, but note GPU nodes
# may lack internet egress (then run --upload_wk separately from htc-el8).

mkdir -p logs

source ~/.bashrc
conda activate trec_seg

# WebKnossos token — keep it OUT of this file (it would end up in git / logs).
# Store it once in ~/.wk_token (chmod 600), or `export WK_TOKEN=...` in your shell
# before `sbatch` (the submitting environment is inherited by default).
[ -z "$WK_TOKEN" ] && [ -f "$HOME/.wk_token" ] && export WK_TOKEN=$(cat "$HOME/.wk_token")

# ── Change this per run ────────────────────────────────────────────────────────
SAMPLE=POR_10to40_20231023_AM_01_epo_02

PRED=/scratch/asvetlove/TREC/predictions/$SAMPLE
RAW=/scratch/asvetlove/TREC/IMATREC_volumes/$SAMPLE
OUT=/scratch/asvetlove/TREC/results/${SAMPLE}_instances

python postprocess.py \
    --input        $PRED \
    --output       $OUT \
    --fg_threshold 0.5 \
    --bd_threshold 0.4 \
    --min_size     500 \
    --upload_wk \
    --raw          $RAW \
    --wk_token     "$WK_TOKEN"
