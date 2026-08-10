#!/bin/bash
#SBATCH --job-name=scgnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=7gb
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --account=chlin
# One job per scenario (4 scenarios -> 0-3); each runs its 20 reps sequentially.
#SBATCH --array=0-3
#SBATCH --output=/scratch/g/chlin/Yushu/results/Comparison_methods/scgnn/logs/%x-%A_%a.out
#SBATCH --mail-type=END
#SBATCH --mail-user=yuswang@mcw.edu
#
#
# Prep genes x cells counts.csv -> PreprocessingscGNN -> scGNN.py per rep, then
# scgnn_eval.py writes results_sim_{i}.npz.

echo "Job started at: $(date)  ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"
module load miniforge
conda activate scgnnEnv
export CUDA_VISIBLE_DEVICES=""

# ----------------- Configuration -----------------
SCGNN_DIR="/scratch/g/chlin/Yushu/scGNN"               # scGNN repo (PreprocessingscGNN.py, scGNN.py)
PREP_PY="/home/yu16889/DMVAE/reproducibility/comparison/prep"             # scgnn_prep.py, scgnn_eval.py
DATA_ROOT="/scratch/g/chlin/Yushu/Data"                # generated sim data (simcounts/simmeta)
RES="/scratch/g/chlin/Yushu/results/Comparison_methods/scgnn"
NREPS=20

SCENARIOS=(
  s01_latent_gmm_k8   # latent Gaussian mixture, K = 8
  s02_latent_gmm_k6   # latent Gaussian mixture, K = 6
  s03_batch_effect    # Splatter, 6 clusters over 3 batches
  s04_mixed_hard      # Splatter, 9 clusters, weak DE
)

S="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
SCGNN_DATA="${SCGNN_DIR}/Data/${S}"        # per-rep sim{i}/counts.csv go here
OUT="${RES}/${S}"                          # per-rep scGNN outputs + results_sim_{i}.npz
mkdir -p "${OUT}/logs"
echo "Scenario: ${S}  (reps 1..${NREPS})"

# ----------------- 1) prep genes x cells counts.csv -----------------
python "${PREP_PY}/scgnn_prep.py" --data-dir "${DATA_ROOT}/${S}" \
  --out-dir "${SCGNN_DATA}" --rep-start 1 --rep-end $((NREPS + 1)) \
  || echo "[WARN] scgnn_prep failed for ${S}"

# ----------------- 2) run scGNN per rep -----------------
cd "${SCGNN_DIR}" || exit 1
for i in $(seq 1 ${NREPS}); do
  DS="sim${i}"; OUTI="${OUT}/${DS}"; mkdir -p "${OUTI}"
  [ -f "${SCGNN_DATA}/${DS}/counts.csv" ] || { echo "  SKIP ${DS}: no counts.csv"; continue; }
  echo "===== ${S}/${DS} ====="
  python -u -W ignore PreprocessingscGNN.py --datasetName counts.csv \
    --datasetDir "${SCGNN_DATA}/${DS}/" --LTMGDir "${SCGNN_DATA}/${DS}/" \
    --filetype CSV --geneSelectnum 2000 || { echo "  [WARN] prep ${DS}"; continue; }
  python -u -W ignore scGNN.py --datasetName "${DS}" --datasetDir "${SCGNN_DATA}/" \
    --outputDir "${OUTI}/" --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --nonsparseMode \
    || echo "  [WARN] scGNN ${DS}"
done

# ----------------- 3) evaluate -> results_sim_{i}.npz (same schema as others) -----------------
python "${PREP_PY}/scgnn_eval.py" --scgnn-out "${OUT}" --label-root "${SCGNN_DATA}" \
  --out-dir "${OUT}" --rep-start 1 --rep-end $((NREPS + 1)) \
  >> "${OUT}/logs/eval.log" 2>&1 || echo "[WARN] scgnn_eval failed for ${S}"

echo "Job ended at: $(date)  (scenario ${S})"
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State 2>/dev/null || true
