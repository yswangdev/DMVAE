# DMVAE

Deep Mixture Variational Auto-Encoder for joint inference of cluster number and
assignment in single-cell RNA-seq data.

DMVAE places a prior over the number of clusters *k* and, conditional on *k*, a
*k*-component Gaussian mixture prior on the latent space of a variational autoencoder.
One training run returns a shared latent embedding, an aggregate support score S(k)
over the candidate range, the cluster number, and cell assignments at **every**
candidate resolution without retraining and without a resolution parameter.

Code for **"DMVAE: Deep Mixture Variational Auto-Encoder for Joint Inference of Cluster
Number and Assignment in Single-Cell RNA-Seq Data."**

## Requirements

Python 3.11, CPU only. The analyses ran on 4–8 cores with up to 32 GB RAM per job.

```bash
pip install -r requirements.txt
```

For an exact core environment matching the verified smoke test, use
`requirements-verified.txt` with Python 3.11.10. The broader `requirements.txt`
retains compatible version ranges for reuse in existing analysis environments.

TensorFlow 2.15 includes the legacy optimizers directly. TensorFlow 2.16+ ships Keras
3, so the legacy optimizers used here need `tf-keras`; `run.py` enables it only for
those newer TensorFlow versions.

R is needed only to reproduce the manuscript: Seurat 5.4.0 and Splatter for the
preprocessing and the simulations.

## Quick start

```bash
python model/run.py \
  --input-datafile /path/to/data --input-file data_norm.txt --meta-file data_celltype.txt \
  --output-base-path /path/to/results \
  --a 2 --b 15
```

`--input-file` is a cell-by-gene matrix, whitespace-separated, no header, min–max
scaled to [0, 1] (see [Input](#input)). `--meta-file` is one integer label per line.
`--a`/`--b` bound the candidate cluster numbers.

Giving a hyperparameter several values turns it into a grid
search; one value each is a single run:

```bash
python model/run.py ... --beta 0.3,0.5,0.8,1.0 --lr-nn 1e-4,1e-5
```

A `{i}` in a filename makes it a series, for simulation replicates:

```bash
python model/run.py ... --input-file simnorm_{i}.txt --meta-file simmeta_{i}.txt --start 1 --end 21
```

Runs are deterministic by default (`--seed 42`); restart `j` uses seed `42 + j`.
Change `--seed` to create an independent run.


## Input

All datasets were preprocessed with one Seurat pipeline
(`reproducibility/preprocess/preprocess.R`): `NormalizeData` (LogNormalize, scale
factor 10<sup>4</sup>), `FindVariableFeatures` (vst, 3,000 genes), `ScaleData`, drop
genes whose scaled values sum to zero, then feature-wise min–max scaling to [0, 1].
The min–max matrix is what DMVAE takes.

## Output

A run writes one file, `dmvae.npz`:

```python
d = np.load("dmvae.npz")
d["embedding"]      # (N, D)   latent means
d["clusters"]       # (N,)     assignment at the selected k, labels 0..K-1
d["S_k"]            # (b-a+1,) aggregate support, aligned to d["k_values"]
d["k_selected"]     # the cluster number
d["n_clusters"]     # components that actually got cells — can be < k_selected
```

`k_map`, `k_knee`, `knee_strength` and `map_at_boundary` are always recorded, so a run
fitted under one selection rule can be re-read under the other without refitting. With
reference labels present, `BestARI`, `BestNMI`, `BestACC` and `labels_true` are
included, along with the per-epoch `ARI`/`NMI`/`K`/`ACC` traces.

| Option | Effect |
|--------|--------|
| `--select map` (default) | k̂ = argmax S(k), the estimator the manuscript reports |
| `--select knee` | k̂ = the knee of S(k), the point furthest from the chord joining the endpoints of the normalised curve. Use when S(k) is multimodal or rises to a boundary, where the maximum reflects the extent of the search rather than support for a model |
| `--multi-resolution` | Adds `clusters_all_k` (N × candidate k) and `umap_2d`, and renders `umap_k{K}.png` for every candidate k |
| `--legacy-artifacts` | Also writes the loose per-quantity files the figure scripts read, plus the loss curves |

The per-k UMAPs carry no ARI, NMI or accuracy annotation: they are the model's output
and read the same whether or not reference labels exist.

A maximum attained at `a` or `b` means the candidate range should be widened, not that
*k* has been estimated. The run prints that warning when it happens.

## Repository layout

```
model/                    the method
  run.py                  command-line entry point
  train.py                one dataset, one hyperparameter set; restarts and artifacts
  model.py                encoder/decoder, the mixture prior p(k), the DMVAE model
  utils.py                autoencoder pretraining, the per-epoch callback, plotting
  evaluation.py           metrics, selecting k from S(k), the dmvae.npz archive
reproducibility/          everything run for the manuscript
  preprocess/             the Seurat pipeline, and the downsampling for the robustness test
  simulation/             the Splatter and latent-GMM generators
  comparison/             wrappers for the five competing methods, and their input prep
  figures/                one entry point per manuscript figure
```

## Reproducing the manuscript

### 0. Preprocessing the real datasets (Methods, *Data preprocessing*)

`reproducibility/preprocess/preprocess.R` is the whole pipeline in one file:
`LogNormalize` at scale factor 10<sup>4</sup>, 3,000 vst HVGs, `ScaleData`, drop genes
whose scaled values sum to zero, then feature-wise min-max scaling to [0, 1]. That
min-max matrix is what DMVAE consumes; the count exports in the same file are what the
competing methods take, since each applies its own preprocessing to counts.

```bash
Rscript preprocess.R              # all 13 benchmark datasets
Rscript preprocess.R Bach Klein   # named datasets only
```

Sourcing the file runs nothing, so the functions can be called directly:

| Function | For |
|----------|-----|
| `prep_seurat_from_h5()` | the published `.h5` datasets (both the dense and CSR layouts appear) |
| `extract_counts_from_h5()` | the raw counts scGNN and scDAC take |
| `prep_pbmc()` | Human PBMC, which has no published annotation and is labelled through the Seurat tutorial's clustering first |
| `prep_seurat_object()` | a Seurat object that already carries its annotation, which is how the CD4 T cell data (GSE310947) ships |

All three entry points share `scale_hvg_minmax()`, so the pipeline is defined once.

The simulation generators carry their own preprocessing, which uses `modelGeneVar` with
500 HVGs instead, as the Methods describe. The notebooks in
`reproducibility/comparison/prep/` are scDAC's own input preparation, not this pipeline.

### 1. Simulated data (Methods, *Simulation design*)

Four scenarios, two from each generator family. Filenames, output folders and
manuscript labels all agree.

| Manuscript | Generator | Design | true k |
|-----------|-----------|--------|--------|
| s01 | `reproducibility/simulation/sim_gen_s01_latent_gmm_k8.r` | latent Gaussian mixture, Poisson lift | 8 |
| s02 | `reproducibility/simulation/sim_gen_s02_latent_gmm_k6.r` | latent Gaussian mixture, Poisson lift | 6 |
| s03 | `reproducibility/simulation/sim_gen_s03_batch_effect.r` | Splatter, 6 clusters over 3 batches, 6,000 cells | 6 |
| s04 | `reproducibility/simulation/sim_gen_s04_mixed_hard.r` | Splatter, 9 clusters with weak DE, 9,000 cells | 9 |

Each generator writes 20 replicates. 

### 2. Comparison methods (Methods, *Benchmarking protocol*)

scVI, scGNN, ADClust, scAce and scDAC are installed from their own repositories and run
with their author-recommended defaults. Only the thin wrappers are here:

| Method | Simulations | Real datasets |
|--------|-------------|---------------|
| scVI | `reproducibility/comparison/simulation/scvi_sim.py` | `reproducibility/comparison/real_world/scvi_rw.py` |
| scAce | `reproducibility/comparison/simulation/scace_sim.py` | `reproducibility/comparison/real_world/scace_rw.py` |
| ADClust | `reproducibility/comparison/simulation/adclust_sim.py` | `reproducibility/comparison/real_world/adclust_rw.py` |
| scGNN | `reproducibility/comparison/prep/scgnn_prep.py` → scGNN → `reproducibility/comparison/prep/scgnn_eval.py` | same |
| scDAC | `reproducibility/comparison/prep/scdac_prep.py` → scDAC `run.py` | same |

The scAce and ADClust wrappers import `reproducibility.utils` from the scAce repository;
put it on `PYTHONPATH`.

### 3. DMVAE runs

Run the grid search directly with `model/run.py`. Comma-separated values create the
Cartesian product of the supplied hyperparameters. 

Use `--select knee` instead of `--select map` to apply the alternative selection rule.
The `--multi-resolution` and `--legacy-artifacts` options are required only when
regenerating the manuscript figures; they may be omitted for an ordinary DMVAE run.

### 4. Robustness (Methods, *Robustness and sensitivity analyses*)

`reproducibility/preprocess/downsample.ipynb` draws the single seed-123 permutation of the Plasschaert
cells and writes the nested 40% ⊂ 60% ⊂ 80% subsets for every method's input format.

### 5. Figures

One entry point per manuscript figure, named for the figure it produces.

| Manuscript | Code |
|-----------|------|
| Fig. 1 (`frameplot.png`) | `fig1.py` — the S(k) bar chart, the illustrative latent space, the p(k) prior |
| Fig. 2 (`sim_plot.png`) | `fig2.py --panel a\|b\|c` |
| Fig. 3a,b (`dmvae.png`) | `fig3ab.py` — also the single source of the 13-dataset ARI / k tables |
| Fig. 3c,d | `fig3cd.ipynb` |
| Fig. 4a,b (`umap_sankey.png`) | `fig4ab.r` — destiny DPT, the pseudotime violins, Wilcoxon brackets |
| Fig. 4c,d | `fig4cd.py` (panel d via `fig4d_sankey.py`) |
| Fig. 5a,b (`cd4.png`) | `fig5ab.py` |
| Fig. 5c–f | `fig5cf.r` — WT/KO composition, marker dot plot, GSEA NES heatmaps |
| Fig. 6 (`rw_fig2.png`) | `fig6.py` |
| Suppl. S1–S3 | `figS1_S3.py --part nmi\|simnmi\|pk\|pkgrid\|extras` |
| Suppl. S4, S5 | `figS4_S5.ipynb` — UMAP grids for the remaining ten datasets |
| Suppl. S7–S9 | `figS7_S9.r` — wide dot plot and full GSEA heatmaps; run `fig5cf.r` first |
| Suppl. S10 | `figS10.py`, `figS10.r` |

All under `reproducibility/figures/`. `fig6.py` and `figS1_S3.py` import the shared
tables from `fig3ab.py`, and `figS7_S9.r` sources `fig5cf.r` for its helpers, so run
them from that directory.

Supplementary Fig. S6 — Seurat clustering of the CD4 cells across resolution 0.2 to 1 —
has no script here; it was produced with the reference pipeline of the original study.

Data and result paths are set at the top of each file, or overridden by `--out-dir` /
environment variables; they still point at the authors' cluster and local paths and must
be repointed.

## Data availability

The 13 benchmark datasets are public; accessions and sources are in Supplementary
Table 1. The CD4 T-cell dataset is GEO **GSE310947**. Simulated data are regenerated from
`R/`.

## License

MIT — see `LICENSE`.
