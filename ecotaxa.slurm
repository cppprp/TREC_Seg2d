#!/bin/bash
#SBATCH --job-name=trec_ecotaxa
#SBATCH --nodes=1
#SBATCH --partition=htc-el8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ecotaxa_%j.log

# CPU node (htc-el8) — needs outbound internet for both WebKnossos and EcoTaxa.
# Instances are streamed from WebKnossos one bounding box at a time, so memory
# stays small regardless of the volume size.

mkdir -p logs

source ~/.bashrc
export PYTHONUNBUFFERED=1

# zarr-3 env (see postprocess.slurm for why the full path is pinned).
PYTHON=/home/asvetlove/miniforge3/envs/trec_seg/bin/python

# Credentials — keep them OUT of this file. Store once (chmod 600) or export
# them in your shell before `sbatch`.
#   ~/.wk_token                         WebKnossos API token
#   ~/.ecotaxa_token                    EcoTaxa JWT, or instead export
#   ECOTAXA_USER / ECOTAXA_PASSWORD     (the script logs in itself)
[ -z "$WK_TOKEN" ]      && [ -f "$HOME/.wk_token" ]      && export WK_TOKEN=$(cat "$HOME/.wk_token")
[ -z "$ECOTAXA_TOKEN" ] && [ -f "$HOME/.ecotaxa_token" ] && export ECOTAXA_TOKEN=$(cat "$HOME/.ecotaxa_token")

# ── Change this per run ────────────────────────────────────────────────────────
SAMPLE=POR_10to40_20231023_AM_01_epo_02
OUT=/scratch/asvetlove/TREC/ecotaxa
PROJECT_ID=        # existing EcoTaxa project id; leave empty to only build the zip

UPLOAD=""
[ -n "$PROJECT_ID" ] && UPLOAD="--upload --project_id $PROJECT_ID"

$PYTHON ecotaxa_export.py \
    --sample     $SAMPLE \
    --output     $OUT \
    --projection sum \
    --min_voxels 500 \
    --workers    $SLURM_CPUS_PER_TASK \
    $UPLOAD
