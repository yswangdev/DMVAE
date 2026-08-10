#!/bin/bash
#SBATCH --job-name=run_all_methods_pbmc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=7gb
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --account=chlin
#SBATCH --output=/scratch/g/chlin/Yushu/results/Comparison_methods/run_all_pbmc/logs/%x-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuswang@mcw.edu
#
# One real dataset, not a simulation: no job array and no rep loop. scAce, ADClust
# and scVI run through the real_world wrappers; scGNN and scDAC through their preps.
#
# Create the log dir before submitting -- Slurm will not.

set -uo pipefail
echo "Job started at: $(date)  JOB_ID=${SLURM_JOB_ID:-NA}"
module load miniforge
export CUDA_VISIBLE_DEVICES=""          # force CPU for every method

# ============================ CONFIG (edit paths) ============================
DS="pbmc_4c"
RW_PY="/home/yu16889/DMVAE/reproducibility/comparison/real_world"   # scace/adclust/scvi _rw.py
PREP_PY="/home/yu16889/DMVAE/reproducibility/comparison/prep"                        # scgnn/scdac prep + scgnn_eval
DATA_ROOT="/scratch/g/chlin/Yushu/Data"                               # holds ${DS}/ (already saved)
RES="/scratch/g/chlin/Yushu/results/Comparison_methods"
X_FILE="pbmc_counts.csv"                                              # raw counts, cells x genes
Y_FILE="pbmc_meta.txt"                                                # labels 1=B 2=CD4/CD8 3=Mono 4=NK
REP="pbmc"                                                            # scGNN/scDAC data subdir name

SCACE_REPRO="/home/yu16889/scAce"                                     # provides reproducibility.utils
ADCLUST_DIR="/scratch/g/chlin/Yushu/ADClust"
SCVI_ENV="/scratch/g/chlin/Yushu/envs/scvi"
SCGNN_DIR="/scratch/g/chlin/Yushu/scGNN"
SCDAC_DIR="/scratch/g/chlin/Yushu/scDAC/scDAC"
SCDAC_ENV="/scratch/g/chlin/Yushu/envs/scdac"
DMVAE_ENV="dmvae"                                                     # conda env for scAce + ADClust
# ===========================================================================

run_block() { echo; echo "===== $* ====="; }

IN="${DATA_ROOT}/${DS}"
mkdir -p "${IN}"
echo; echo "##################### DATASET: ${DS} (real, single dataset) #####################"

# ---- 0) input check ----
for f in "${X_FILE}" "${Y_FILE}"; do
  [[ -e "${IN}/${f}" ]] || { echo "Missing ${IN}/${f} -- saved PBMC dataset not found" >&2; exit 2; }
done

# ---- 1) scAce (env: dmvae; needs reproducibility module) ----
run_block "[1] scAce ${DS}"
conda activate "${DMVAE_ENV}"
export PYTHONPATH="${PYTHONPATH:-}:${SCACE_REPRO}"
mkdir -p "${RES}/scace/${DS}/logs"
python "${RW_PY}/scace_rw.py" --data-dir "${IN}/" --out-dir "${RES}/scace/${DS}/" \
--expression-file "${X_FILE}" --labels-file "${Y_FILE}" \
>> "${RES}/scace/${DS}/logs/out.log" 2>&1 || echo "[WARN] scAce failed ${DS}"

# ---- 2) ADClust (env: dmvae; dip.so on LD path) ----
run_block "[2] ADClust ${DS}"
conda activate "${DMVAE_ENV}"
export PYTHONPATH="${PYTHONPATH:-}:${ADCLUST_DIR}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ADCLUST_DIR}:${ADCLUST_DIR}/dip.so"
mkdir -p "${RES}/adclust/${DS}/logs"
( cd "${ADCLUST_DIR}" && python "${RW_PY}/adclust_rw.py" --data-dir "${IN}/" \
--out-dir "${RES}/adclust/${DS}/" --expression-file "${X_FILE}" --labels-file "${Y_FILE}" ) \
>> "${RES}/adclust/${DS}/logs/out.log" 2>&1 || echo "[WARN] ADClust failed ${DS}"

# ---- 3) scVI (env: scvi) ----
run_block "[3] scVI ${DS}"
conda activate "${SCVI_ENV}"
export PYTHONPATH="${PYTHONPATH:-}:${SCACE_REPRO}"   # for reproducibility.utils (set_seed/metric)
mkdir -p "${RES}/scvi/${DS}/logs"
python "${RW_PY}/scvi_rw.py" --data-dir "${IN}/" --out-dir "${RES}/scvi/${DS}/" \
--expression-file "${X_FILE}" --labels-file "${Y_FILE}" \
>> "${RES}/scvi/${DS}/logs/out.log" 2>&1 || echo "[WARN] scVI failed ${DS}"

# ---- 4) scGNN (env: scgnnEnv) ----
run_block "[4] scGNN ${DS}"
conda activate scgnnEnv
SCGNN_DATA="${SCGNN_DIR}/Data/${DS}"
python "${PREP_PY}/scgnn_prep.py" --data-dir "${IN}" --out-dir "${SCGNN_DATA}" \
--counts-file "${X_FILE}" --labels-file "${Y_FILE}" --rep-name "${REP}" \
|| echo "[WARN] scGNN prep failed ${DS}"
mkdir -p "${RES}/scgnn/${DS}/logs"
OUTI="${RES}/scgnn/${DS}/${REP}"; mkdir -p "${OUTI}"
( cd "${SCGNN_DIR}" || exit 1
  [ -f "${SCGNN_DATA}/${REP}/counts.csv" ] || { echo "missing ${SCGNN_DATA}/${REP}/counts.csv"; exit 1; }
  python -u -W ignore PreprocessingscGNN.py --datasetName counts.csv \
    --datasetDir "${SCGNN_DATA}/${REP}/" --LTMGDir "${SCGNN_DATA}/${REP}/" \
    --filetype CSV --geneSelectnum 2000
  python -u -W ignore scGNN.py --datasetName "${REP}" --datasetDir "${SCGNN_DATA}/" \
    --outputDir "${OUTI}/" --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --nonsparseMode \
) >> "${RES}/scgnn/${DS}/logs/out.log" 2>&1 || echo "[WARN] scGNN run failed ${DS}"
python "${PREP_PY}/scgnn_eval.py" --scgnn-out "${RES}/scgnn/${DS}" --label-root "${SCGNN_DATA}" \
--out-dir "${RES}/scgnn/${DS}" --rep-name "${REP}" \
>> "${RES}/scgnn/${DS}/logs/eval.log" 2>&1 || echo "[WARN] scGNN eval failed ${DS}"

# ---- 5) scDAC (env: scdac) ----
# --n-hvg 4000 selects here (13714 genes). That path calls scanpy
# highly_variable_genes(flavor="seurat_v3"), which needs scanpy and scikit-misc
# in the scdac env.
run_block "[5] scDAC ${DS}"
conda activate "${SCDAC_ENV}"
SCDAC_DATA="${SCDAC_DIR}/data/${DS}"
python "${PREP_PY}/scdac_prep.py" --data-dir "${IN}" --out-dir "${SCDAC_DATA}" \
--rel-prefix "data/${DS}" --data-toml "${SCDAC_DIR}/configs/data.toml" \
--counts-file "${X_FILE}" --labels-file "${Y_FILE}" --rep-name "${REP}" --task-name "${DS}" \
--n-hvg 4000 --N 512 || echo "[WARN] scDAC prep failed ${DS}"
mkdir -p "${RES}/scdac/${DS}/logs"
( cd "${SCDAC_DIR}" || exit 1
  [ -d "${SCDAC_DATA}/${REP}" ] || { echo "missing ${SCDAC_DATA}/${REP}"; exit 1; }
  python -u run.py --action train_and_infer --task "${DS}" --exp e0 --model default \
    --data-dir "${SCDAC_DATA}/${REP}/" --label-path "${SCDAC_DATA}/${REP}/label.csv" \
) >> "${RES}/scdac/${DS}/logs/out.log" 2>&1 || echo "[WARN] scDAC run failed ${DS}"

echo; echo "Job ended at: $(date)  (dataset ${DS})"
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,Elapsed,MaxRSS,State 2>/dev/null || true
