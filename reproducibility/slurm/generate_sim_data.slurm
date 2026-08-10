#!/bin/bash
#SBATCH --job-name=gen_sim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=7gb
#SBATCH --partition=normal
#SBATCH --time=6:00:00
#SBATCH --account=chlin
#SBATCH --array=0-3
#SBATCH --output=/scratch/g/chlin/Yushu/Data/_logs/gen-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuswang@mcw.edu
#
#
# One array task per R generator; each parallelises its 20 reps internally and
# writes into ${DATA_ROOT}/<scenario>/ (redirected via SIM_OUT_ROOT).

set -uo pipefail
echo "Job started at: $(date)  ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"

module load R/4.4.2

# ----------------------------- CONFIG -----------------------------
REPO_R="${REPO_R:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../simulation" && pwd)}"   # dir holding the sim_gen_*.r generators
DATA_ROOT="/scratch/g/chlin/Yushu/Data"        # all scenarios written here
mkdir -p "${DATA_ROOT}/_logs"

# All splatter generators (array index -> file). Edit to add/remove scenarios.
GENERATORS=(
  sim_gen_s01_latent_gmm_k8.r   # latent Gaussian mixture, K = 8
  sim_gen_s02_latent_gmm_k6.r   # latent Gaussian mixture, K = 6
  sim_gen_s03_batch_effect.r    # Splatter, 6 clusters over 3 batches
  sim_gen_s04_mixed_hard.r      # Splatter, 9 clusters, weak DE
)
# ------------------------------------------------------------------

GEN="${GENERATORS[$SLURM_ARRAY_TASK_ID]}"
echo "Generator: ${GEN}"
echo "Output root: ${DATA_ROOT}"

SIM_OUT_ROOT="${DATA_ROOT}" Rscript "${REPO_R}/${GEN}"
status=$?

echo "Rscript exit status: ${status}"
echo "Job ended at: $(date)"
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,Elapsed,MaxRSS,State 2>/dev/null || true
exit ${status}
