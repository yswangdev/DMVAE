#!/bin/bash
# CPU-only DMVAE grid search (no GPU). Use tensorflow-cpu + tf-keras in conda env `dmvae`.
#SBATCH --job-name=dmvae-grid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --time=100:00:00
#SBATCH --account=chlin
#SBATCH --output=/scratch/g/chlin/Yushu/results/log/%x-%A_%a.out
#SBATCH --mail-type=END
#SBATCH --mail-user=yuswang@mcw.edu
# One array task per scenario (4 scenarios -> 0-3).
#SBATCH --array=0-3

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)"

module load miniforge
conda activate dmvae
# Recommended for CPU-only: pip install tensorflow-cpu tf-keras  (uninstall tensorflow first if present)

# Optional: ensure package is importable (editable install in env)
# export PYTHONPATH="/home/yu16889/DMVAE/DMVAE-run/src:${PYTHONPATH:-}"

# ---------------- DATASETS (array index) --------------------------------
# One scenario per array index; folder names match the generator outputs under DATA_BASE.
DATASETS=(
    s01_latent_gmm_k8              # 0  latent Gaussian mixture, K = 8
    s02_latent_gmm_k6              # 1  latent Gaussian mixture, K = 6
    s03_batch_effect               # 2  Splatter, 6 clusters over 3 batches
    s04_mixed_hard                 # 3  Splatter, 9 clusters, weak DE
)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# DMVAE K-search range per scenario: [A, B] = [truth_k - 2, truth_k + 2] (A clamped to >= 2)
declare -A A_MAP
declare -A B_MAP
A_MAP[s01_latent_gmm_k8]=6           ; B_MAP[s01_latent_gmm_k8]=10
A_MAP[s02_latent_gmm_k6]=4           ; B_MAP[s02_latent_gmm_k6]=8
A_MAP[s03_batch_effect]=4            ; B_MAP[s03_batch_effect]=8
A_MAP[s04_mixed_hard]=7              ; B_MAP[s04_mixed_hard]=11

A_DEFAULT=2
B_DEFAULT=15

A_MIN="${A_MAP[$DS]:-$A_DEFAULT}"
B_MAX="${B_MAP[$DS]:-$B_DEFAULT}"

# --------------- GRID ---------------
# beta, lr_nn (lr_gmm follows lr_nn), ae_lr, ae_epoch: 2 x 2 x 3 x 3 = 36 combos.
AE_LR_GRID="1e-3,1e-4"          # AE pretrain LR: fast vs safe
AE_EPOCH_GRID="20,40"           # AE pretrain epochs (10 is usually too few)
LR_NN_GRID="1e-4,1e-5,1e-6"     # DMVAE joint VAE+GMM LR (incl. conservative default)
BETA_GRID="0.1,0.5,1"           # KL weight: weak / moderate / full ELBO

# --------------- PATHS ----------------------------------------------------
# Entry point; override to point at another checkout.
DMVAE_RUN="${DMVAE_RUN:-python /home/yu16889/DMVAE/model/run.py}"

DATA_BASE="/scratch/g/chlin/Yushu/Data"
RES_BASE="/scratch/g/chlin/Yushu/results/dmvae/"

INPUT_DIR="${DATA_BASE}/${DS}/"
RESULTS_BASE="${RES_BASE}/${DS}/grid_$(date +%Y%m%d)"
GRID_BASE="${RESULTS_BASE}"
mkdir -p "${GRID_BASE}"
mkdir -p "/scratch/g/chlin/Yushu/results/log"

# --------------- DEFAULT FILENAMES --------------------------------------
# Simulation export naming (splatter / latent-GMM generators); override per-dataset below.
INPUT_FILE="simnorm_1.txt"
META_FILE="simmeta_1.txt"

# --------------- TRUTH K --------------------------------------------------
declare -A TRUTHK
TRUTHK[s01_latent_gmm_k8]=8
TRUTHK[s02_latent_gmm_k6]=6
TRUTHK[s03_batch_effect]=6
TRUTHK[s04_mixed_hard]=9

TRUTH_K_FLAG=()
if [[ -n "${TRUTHK[${DS}]+set}" && -n "${TRUTHK[${DS}]}" ]]; then
  TRUTH_K_FLAG=(--reference-k "${TRUTHK[${DS}]}")
elif [[ -f "${INPUT_DIR}/truth_k.txt" ]]; then
  TK=$(tr -d ' \t\r\n' < "${INPUT_DIR}/truth_k.txt")
  if [[ "$TK" =~ ^[0-9]+$ ]]; then TRUTH_K_FLAG=(--reference-k "$TK"); fi
fi

# --------------- SANITY CHECKS -------------------------------------------
echo "Dataset: ${DS}"
echo "Input dir: ${INPUT_DIR}"
echo "Results: ${RESULTS_BASE}"
echo "Files: ${INPUT_FILE} | ${META_FILE}"
[[ -f "${INPUT_DIR}/${INPUT_FILE}" ]] || { echo "Missing ${INPUT_DIR}/${INPUT_FILE}" >&2; exit 2; }
[[ -f "${INPUT_DIR}/${META_FILE}" ]]  || { echo "Missing ${INPUT_DIR}/${META_FILE}"  >&2; exit 2; }

# Keras 2 / tf_keras (TensorFlow 2.16+)
export TF_USE_LEGACY_KERAS=1
# HPC stability: TensorFlow + OpenMP/MKL on shared nodes (tune as needed)
# oneDNN custom ops segfault on these CPU nodes: force off (override any inherited =1).
export TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
# CPU-only: do not use CUDA (TensorFlow sees no GPUs; avoids driver/CUDA crashes on cpu nodes)
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# --------------- RUN ---------------
# --multi-resolution and --legacy-artifacts are set because the figure scripts need
# the per-k assignments and the loose files. A plain run needs neither.
SELECT_RULE="${SELECT_RULE:-map}"

${DMVAE_RUN} \
  --input-datafile "${INPUT_DIR}" \
  --input-file "${INPUT_FILE}" \
  --meta-file "${META_FILE}" \
  --output-base-path "${GRID_BASE}" \
  --a "${A_MIN}" \
  --b "${B_MAX}" \
  --epochs 200 \
  --ae-lr "${AE_LR_GRID}" \
  --ae-epoch "${AE_EPOCH_GRID}" \
  --lr-nn "${LR_NN_GRID}" \
  --beta "${BETA_GRID}" \
  --select "${SELECT_RULE}" \
  --multi-resolution \
  --legacy-artifacts \
  "${TRUTH_K_FLAG[@]}"

echo "Job finished: $(date)"
