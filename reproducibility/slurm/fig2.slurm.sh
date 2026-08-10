#!/bin/bash
# Render Figure 2 (simulation panels) from the saved DMVAE / comparison results.
# Read-only with respect to the result files: it only loads .npz / .txt and writes images.
#SBATCH --job-name=fig2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --account=chlin
#SBATCH --output=/scratch/g/chlin/Yushu/results/log/%x-%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=yuswang@mcw.edu
#
# Usage
#   sbatch fig2.slurm.sh                 # all panels
#   sbatch fig2.slurm.sh a               # ARI boxplots
#   sbatch fig2.slurm.sh b               # cascading Sankey
#   sbatch fig2.slurm.sh c               # 2x2 UMAP grid
#
# Runs interactively too:
#   bash fig2.slurm.sh a

set -euo pipefail

PANEL="${1:-all}"

FIG_DIR="${FIG_DIR:-/home/yu16889/DMVAE/reproducibility/figures}"
OUT_DIR="${OUT_DIR:-/scratch/g/chlin/Yushu/results/dmvae/figures}"

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Panel: ${PANEL}"
echo "Figures dir: ${FIG_DIR}"
echo "Output dir: ${OUT_DIR}"

module load miniforge
conda activate scdac

mkdir -p "${OUT_DIR}"
cd "${FIG_DIR}"

python fig2.py --panel "${PANEL}" --out-dir "${OUT_DIR}"

echo "Wrote to: ${OUT_DIR}"
ls -la "${OUT_DIR}"
echo "Job finished at: $(date)"
