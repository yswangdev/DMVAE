#!/bin/bash
#SBATCH --job-name=run_all_methods
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=7gb
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --account=chlin
# One array task PER SCENARIO (4 scenarios -> 0-3). Set last index = #scenarios - 1.
#SBATCH --array=0-3
#SBATCH --output=/scratch/g/chlin/Yushu/results/Comparison_methods/run_all/logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuswang@mcw.edu
#
# One array task per scenario: run the 5 comparison methods, each in its own conda
# env. scAce / ADClust / scVI read the raw-count h5 (sim_{i}.h5); scGNN reads
# genes x cells counts.csv; scDAC reads its 4000-HVG data dir.

set -uo pipefail
echo "Job started at: $(date)  JOB_ID=${SLURM_JOB_ID:-NA}"
module load miniforge
export CUDA_VISIBLE_DEVICES=""          # force CPU for every method

# ============================ CONFIG (edit paths) ============================
REPO_R="/home/yu16889/DMVAE/reproducibility/simulation"                                      # R generators (sim_gen_*.r)
SIM_PY="/home/yu16889/DMVAE/reproducibility/comparison/simulation"  # scace/adclust/scvi _sim.py
# comparison/simulation/ holds the 20-rep wrappers; comparison/real_world/ the
# single-dataset counterparts used for the 13 real benchmarks.
PREP_PY="/home/yu16889/DMVAE/reproducibility/comparison/prep"                          # scgnn/scdac prep + scgnn_eval
DATA_ROOT="/scratch/g/chlin/Yushu/Data"                         # generated data per scenario
RES="/scratch/g/chlin/Yushu/results/Comparison_methods"        # results per method/scenario
NSTART=1; NEND=21                                               # rep range [start, end)

SCACE_REPRO="/home/yu16889/scAce"                              # provides reproducibility.utils
ADCLUST_DIR="/scratch/g/chlin/Yushu/ADClust"
SCVI_ENV="/scratch/g/chlin/Yushu/envs/scvi"
SCGNN_DIR="/scratch/g/chlin/Yushu/scGNN"
SCDAC_DIR="/scratch/g/chlin/Yushu/scDAC/scDAC"
DMVAE_ENV="dmvae"                                              # conda env for scAce + ADClust python
R_MODULE="R/4.4.2"                                             # module providing R with splatter/scran/scater/rhdf5

# scenario -> R generator (extend with every splatter sim_gen_* as needed)
declare -A GEN=(
  [s01_latent_gmm_k8]=sim_gen_s01_latent_gmm_k8.r
  [s02_latent_gmm_k6]=sim_gen_s02_latent_gmm_k6.r
  [s03_batch_effect]=sim_gen_s03_batch_effect.r
  [s04_mixed_hard]=sim_gen_s04_mixed_hard.r
)
SCENARIOS=(
  s01_latent_gmm_k8   # latent Gaussian mixture, K = 8
  s02_latent_gmm_k6   # latent Gaussian mixture, K = 6
  s03_batch_effect    # Splatter, 6 clusters over 3 batches
  s04_mixed_hard      # Splatter, 9 clusters, weak DE
)
# ===========================================================================

run_block() { echo; echo "===== $* ====="; }

# This array task handles exactly one scenario.
S="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
echo; echo "##################### SCENARIO: ${S} (array ${SLURM_ARRAY_TASK_ID}) #####################"
IN="${DATA_ROOT}/${S}"
mkdir -p "${IN}"

# ---- 0) generate data: disabled here; use generate_sim_data.slurm ----

# ---- 1) scAce (env: dmvae; needs reproducibility module) ----
run_block "[1] scAce ${S}"
conda activate "${DMVAE_ENV}"
export PYTHONPATH="${PYTHONPATH:-}:${SCACE_REPRO}"
mkdir -p "${RES}/scace/${S}/logs"
python "${SIM_PY}/scace_sim.py" --data-dir "${IN}/" --out-dir "${RES}/scace/${S}/" \
--rep-start ${NSTART} --rep-end ${NEND} >> "${RES}/scace/${S}/logs/out.log" 2>&1 || echo "[WARN] scAce failed ${S}"

# ---- 2) ADClust (env: dmvae; dip.so on LD path) ----
run_block "[2] ADClust ${S}"
conda activate "${DMVAE_ENV}"
export PYTHONPATH="${PYTHONPATH:-}:${ADCLUST_DIR}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ADCLUST_DIR}:${ADCLUST_DIR}/dip.so"
mkdir -p "${RES}/adclust/${S}/logs"
( cd "${ADCLUST_DIR}" && python "${SIM_PY}/adclust_sim.py" --data-dir "${IN}/" \
--out-dir "${RES}/adclust/${S}/" --rep-start ${NSTART} --rep-end ${NEND} ) \
>> "${RES}/adclust/${S}/logs/out.log" 2>&1 || echo "[WARN] ADClust failed ${S}"

# ---- 3) scVI (env: scvi) ----
run_block "[3] scVI ${S}"
conda activate "${SCVI_ENV}"
export PYTHONPATH="${PYTHONPATH:-}:${SCACE_REPRO}"   # for reproducibility.utils (set_seed/metric)
mkdir -p "${RES}/scvi/${S}/logs"
python "${SIM_PY}/scvi_sim.py" --data-dir "${IN}/" --out-dir "${RES}/scvi/${S}/" \
--rep-start ${NSTART} --rep-end ${NEND} >> "${RES}/scvi/${S}/logs/out.log" 2>&1 || echo "[WARN] scVI failed ${S}"

# ---- 4) scGNN (env: scgnnEnv) ----
run_block "[4] scGNN ${S}"
conda activate scgnnEnv
SCGNN_DATA="${SCGNN_DIR}/Data/${S}"
python "${PREP_PY}/scgnn_prep.py" --data-dir "${IN}" --out-dir "${SCGNN_DATA}" \
--rep-start ${NSTART} --rep-end ${NEND} || echo "[WARN] scGNN prep failed ${S}"
mkdir -p "${RES}/scgnn/${S}/logs"
( cd "${SCGNN_DIR}" || exit 1
  for i in $(seq ${NSTART} $((NEND-1))); do
    DS="sim${i}"; OUTI="${RES}/scgnn/${S}/${DS}"; mkdir -p "${OUTI}"
    [ -f "${SCGNN_DATA}/${DS}/counts.csv" ] || continue
    python -u -W ignore PreprocessingscGNN.py --datasetName counts.csv \
      --datasetDir "${SCGNN_DATA}/${DS}/" --LTMGDir "${SCGNN_DATA}/${DS}/" \
      --filetype CSV --geneSelectnum 2000
    python -u -W ignore scGNN.py --datasetName "${DS}" --datasetDir "${SCGNN_DATA}/" \
      --outputDir "${OUTI}/" --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --nonsparseMode
  done ) >> "${RES}/scgnn/${S}/logs/out.log" 2>&1 || echo "[WARN] scGNN run failed ${S}"
python "${PREP_PY}/scgnn_eval.py" --scgnn-out "${RES}/scgnn/${S}" --label-root "${SCGNN_DATA}" \
--out-dir "${RES}/scgnn/${S}" --rep-start ${NSTART} --rep-end ${NEND} \
>> "${RES}/scgnn/${S}/logs/eval.log" 2>&1 || echo "[WARN] scGNN eval failed ${S}"

# ---- 5) scDAC: disabled here; use scdac.slurm ----

echo; echo "Job ended at: $(date)  (scenario ${S})"
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,Elapsed,MaxRSS,State 2>/dev/null || true
