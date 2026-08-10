#!/bin/bash
#SBATCH --job-name=scdac
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=7gb
#SBATCH --partition=normal
#SBATCH --time=72:00:00
#SBATCH --account=chlin
# One job per scenario (4 scenarios -> 0-3); each job runs its 20 reps sequentially.
#SBATCH --array=0-3
#SBATCH --output=/scratch/g/chlin/Yushu/results/Comparison_methods/scdac/logs/%x-%A_%a.out
#SBATCH --mail-type=END
#SBATCH --mail-user=yuswang@mcw.edu
#
#
# Each array task preps its scenario inline (scdac_prep.py flock-protects the shared
# configs/data.toml) then runs all reps.

echo "Job started at: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"

module load miniforge
conda activate /scratch/g/chlin/Yushu/envs/scdac
export CUDA_VISIBLE_DEVICES=""        # CPU env

# -----------------------
# Configuration
# -----------------------
PROJECT_DIR="/scratch/g/chlin/Yushu/scDAC/scDAC"
PREP_PY="/home/yu16889/DMVAE/reproducibility/comparison/prep"        # scdac_prep.py
DATA_ROOT="/scratch/g/chlin/Yushu/Data"           # generated sim data (simcounts/simmeta)
EXP="e0"
MODEL="default"
NREPS=20

# Must match the scenarios prepared above (order defines the array mapping).
SCENARIOS=(
  s01_latent_gmm_k8   # latent Gaussian mixture, K = 8
  s02_latent_gmm_k6   # latent Gaussian mixture, K = 6
  s03_batch_effect    # Splatter, 6 clusters over 3 batches
  s04_mixed_hard      # Splatter, 9 clusters, weak DE
)

# -----------------------
# One scenario per array task; loop its NREPS reps sequentially
# -----------------------
S="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
echo "Scenario: ${S}  (reps 1..${NREPS})"
cd "${PROJECT_DIR}" || exit 1

# --- prep this scenario's tasks (flock-safe data.toml registration) ---
python "${PREP_PY}/scdac_prep.py" \
  --data-dir "${DATA_ROOT}/${S}" \
  --out-dir "${PROJECT_DIR}/data/${S}" \
  --rel-prefix "data/${S}" \
  --data-toml "${PROJECT_DIR}/configs/data.toml" \
  --task-prefix "${S}_sim" --n-hvg 4000 --N 512 \
  --rep-start 1 --rep-end $((NREPS + 1)) || echo "[WARN] scdac_prep failed for ${S}"

fail=0
for ri in $(seq 1 ${NREPS}); do
  TASK="${S}_sim${ri}"                        # matches scdac_prep --task-prefix
  DATA_DIR="${PROJECT_DIR}/data/${S}/sim${ri}"
  LABEL="${DATA_DIR}/label.csv"
  echo "===== ${TASK}  (data-dir=${DATA_DIR}) ====="
  if [ ! -d "${DATA_DIR}" ]; then
    echo "  SKIP: ${DATA_DIR} not found — run scdac_prep (see PREREQUISITE header) first." >&2
    fail=1; continue
  fi
  python -u run.py \
    --action train_and_infer \
    --task "${TASK}" \
    --exp "${EXP}" \
    --model "${MODEL}" \
    --data-dir "${DATA_DIR}/" \
    --label-path "${LABEL}" || { echo "  [WARN] ${TASK} failed"; fail=1; }
done

echo "Job ended at: $(date)  (fail flag=${fail})"
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State 2>/dev/null || true
exit ${fail}
