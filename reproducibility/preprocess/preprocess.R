#!/usr/bin/env Rscript
#
# Data preprocessing for the real datasets (Methods, "Data preprocessing").
#
# Three entry points -- .h5, a 10x directory, an in-memory Seurat object -- sharing
# one pipeline in scale_hvg_minmax():
#
#   NormalizeData(LogNormalize, 1e4) -> FindVariableFeatures(vst, nfeatures)
#     -> ScaleData -> drop zero-sum genes -> per-gene min-max to [0, 1]
#
# The min-max matrix is DMVAE's input; the count exports are what the competing
# methods take.
#
# Usage:
#   Rscript preprocess.R                 # all 13 benchmark datasets
#   Rscript preprocess.R Bach Klein      # named datasets only
#
# Sourcing defines the functions without running the driver.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

# The 13 real benchmark datasets (Supplementary Table 1), as folders under DATA_ROOT.
BENCHMARK_DATASETS <- c(
  "Bach",
  "human_p",
  "Klein",
  "mouse_h",
  "Muraro",
  "PBMC",
  "Plasschaert",
  "Quake_Smart-seq2_Limb_Muscle",
  "Quake_Smart-seq2_Trachea",
  "Romanov",
  "turtle_b",
  "Wang_Lung",
  "Young"
)

DATA_ROOT <- Sys.getenv("DATA_ROOT", unset = "/Volumes/SSD/MCW/Research/Aim 1/Data")

# These ship as a dense X rather than the CSR layout.
DENSE_DATASETS <- c("human_p", "human_k")


# =========================================================================== #
# 1. The shared pipeline
# =========================================================================== #

minmax_01 <- function(x) {
  # Constant genes have zero range; the manuscript sets them to zero.
  rng <- range(x, finite = TRUE)
  if (!all(is.finite(rng)) || diff(rng) == 0) return(rep(0, length(x)))
  (x - rng[1]) / diff(rng)
}


scale_hvg_minmax <- function(so,
                             assay = "RNA",
                             scale_factor = 10000,
                             nfeatures = 3000,
                             scale_features = c("hvg", "all"),
                             drop_zero_sum_genes = TRUE,
                             do_minmax = TRUE,
                             verbose = FALSE) {
  # Returns the cells x genes matrix DMVAE consumes and the Seurat object it came
  # from. scale_features = "hvg" scales only the variable features; "all" scales
  # every gene, which the PBMC path needs for its PCA and clustering.
  scale_features <- match.arg(scale_features)

  so <- Seurat::NormalizeData(so, assay = assay, normalization.method = "LogNormalize",
                              scale.factor = scale_factor, verbose = verbose)
  so <- Seurat::FindVariableFeatures(so, assay = assay, selection.method = "vst",
                                     nfeatures = nfeatures, verbose = verbose)

  hvgs <- Seurat::VariableFeatures(so)
  features <- if (scale_features == "all") rownames(so) else hvgs
  so <- Seurat::ScaleData(so, assay = assay, features = features, verbose = verbose)

  # scale.data is genes x cells; the model wants cells x genes.
  scale_mat <- Seurat::GetAssayData(so, assay = assay, slot = "scale.data")
  scale_mat <- scale_mat[hvgs, , drop = FALSE]
  X <- t(as.matrix(scale_mat))

  if (drop_zero_sum_genes) {
    keep <- colSums(X) != 0
    X <- X[, keep, drop = FALSE]
  }

  if (do_minmax) {
    genes <- colnames(X)
    cells <- rownames(X)
    X <- apply(X, 2, minmax_01)
    X <- as.matrix(X)
    colnames(X) <- genes
    rownames(X) <- cells
  }

  list(so = so, hvgs = hvgs, norm = as.data.frame(X))
}


write_dmvae_input <- function(norm_df, labels, outfile_norm, outfile_celltype = NULL) {
  # The min-max matrix and one integer label per cell, both headerless.
  if (is.null(outfile_celltype)) {
    outfile_celltype <- sub("norm\\.txt$", "celltype.txt", outfile_norm)
    if (identical(outfile_celltype, outfile_norm)) {
      outfile_celltype <- paste0(outfile_norm, "_celltype.txt")
    }
  }
  out_dir <- dirname(outfile_norm)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  write.table(norm_df, outfile_norm, sep = "\t", quote = FALSE,
              col.names = FALSE, row.names = FALSE)
  write.table(labels, outfile_celltype, sep = "\t", quote = FALSE,
              col.names = FALSE, row.names = FALSE)
  invisible(list(norm = outfile_norm, celltype = outfile_celltype))
}


write_scdac_layout <- function(count_data, labels, out_dir) {
  # The CSV layout scdac_prep.py turns into scDAC's per-cell vectors.
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  feat_dir <- file.path(out_dir, "feat")
  if (!dir.exists(feat_dir)) dir.create(feat_dir, recursive = TRUE)

  write.csv(count_data, file.path(out_dir, "rna.csv"), quote = FALSE)
  write.csv(colnames(count_data), file.path(out_dir, "feat_name.csv"),
            quote = FALSE, row.names = FALSE)
  write.csv(rownames(count_data), file.path(out_dir, "cell_name.csv"),
            quote = FALSE, row.names = FALSE)

  feat_dims <- data.frame(rna = ncol(count_data))
  rownames(feat_dims) <- 1
  write.csv(feat_dims, file.path(feat_dir, "feat_dims.csv"), quote = FALSE)

  write.csv(data.frame(cell = seq_along(labels) - 1, label = labels),
            file.path(out_dir, "label.csv"), row.names = FALSE, quote = FALSE)
}


# =========================================================================== #
# 2. Reading the published .h5 files
# =========================================================================== #

read_counts_h5 <- function(h5_path, fname) {
  # Returns a genes x cells dgCMatrix and the cell labels. Two on-disk layouts
  # appear across the benchmark: a dense X, and the CSR triplet under exprs/.
  stopifnot(file.exists(h5_path))
  suppressPackageStartupMessages(library(rhdf5))

  if (fname %in% DENSE_DATASETS) {
    message("Using dense layout for ", fname)
    h <- rhdf5::H5Fopen(h5_path)
    on.exit(rhdf5::H5Fclose(h), add = TRUE)

    X <- h$X
    celltype <- tryCatch(h$Y, error = function(e) NULL)
    counts <- if (nrow(X) >= ncol(X)) X else t(X)
  } else {
    message("Using CSR reconstruction for ", fname)
    x   <- rhdf5::h5read(h5_path, "exprs/data")
    j   <- rhdf5::h5read(h5_path, "exprs/indices")
    p   <- rhdf5::h5read(h5_path, "exprs/indptr")
    shp <- rhdf5::h5read(h5_path, "exprs/shape")   # c(n_cells, n_genes)

    n_cells <- as.integer(shp[1])
    n_genes <- as.integer(shp[2])
    counts <- Matrix::sparseMatrix(
      i = as.integer(j) + 1,                       # 0-based to 1-based; rows = genes
      j = rep.int(seq_len(n_cells), diff(p)),      # cols = cells
      x = as.numeric(x),
      dims = c(n_genes, n_cells)
    )
    celltype <- tryCatch(rhdf5::h5read(h5_path, "obs/cell_type1"), error = function(e) NULL)
  }

  obs_names <- tryCatch(rhdf5::h5read(h5_path, "obs_names"), error = function(e) NULL)
  var_names <- tryCatch(rhdf5::h5read(h5_path, "var_names"), error = function(e) NULL)

  if (!is.null(var_names) && length(var_names) == nrow(counts)) {
    rownames(counts) <- make.unique(as.character(var_names))
  } else if (is.null(rownames(counts)) || anyNA(rownames(counts))) {
    rownames(counts) <- paste0("gene_", seq_len(nrow(counts)))
  }
  if (!is.null(obs_names) && length(obs_names) == ncol(counts)) {
    colnames(counts) <- as.character(obs_names)
  } else if (is.null(colnames(counts)) || anyNA(colnames(counts))) {
    colnames(counts) <- paste0("cell_", seq_len(ncol(counts)))
  }

  if (!inherits(counts, "dgCMatrix")) {
    counts <- as(as.matrix(counts), "dgCMatrix")
  }
  if (!is.null(celltype) && length(celltype) == ncol(counts)) {
    names(celltype) <- colnames(counts)
  }

  list(counts = counts, celltype = celltype)
}


prep_seurat_from_h5 <- function(h5_path,
                                fname,
                                outfile = NULL,
                                assay = "RNA",
                                min_cells = 3,
                                min_features = 200,
                                scale_factor = 10000,
                                nfeatures = 3000,
                                drop_zero_sum_genes = TRUE,
                                do_minmax = TRUE,
                                scdac_dir = NULL) {
  # The .h5 entry point: the model input for one benchmark dataset, and optionally
  # the scDAC layout for the same cells.
  loaded <- read_counts_h5(h5_path, fname)

  so <- Seurat::CreateSeuratObject(counts = loaded$counts,
                                   min.cells = min_cells,
                                   min.features = min_features)
  so$label <- as.factor(loaded$celltype[colnames(so)])

  res <- scale_hvg_minmax(so, assay = assay, scale_factor = scale_factor,
                          nfeatures = nfeatures, scale_features = "hvg",
                          drop_zero_sum_genes = drop_zero_sum_genes,
                          do_minmax = do_minmax)
  labels <- as.numeric(res$so$label)
  stopifnot(length(labels) == nrow(res$norm))

  if (!is.null(outfile)) {
    write_dmvae_input(res$norm, labels, outfile)
  }
  if (!is.null(scdac_dir)) {
    count_data <- t(as.matrix(
      Seurat::GetAssayData(res$so, assay = assay, slot = "counts")[res$hvgs, ]
    ))
    write_scdac_layout(count_data, labels, scdac_dir)
  }

  invisible(list(norm = res$norm, celltype = labels, hvgs = res$hvgs, so = res$so))
}


extract_counts_from_h5 <- function(h5_path, fname,
                                   outfile = NULL,
                                   save_counts_csv = FALSE) {
  # Raw counts for the methods that do their own preprocessing. Writes the label
  # table beside outfile; the dense counts CSV only on request.
  loaded <- read_counts_h5(h5_path, fname)
  counts <- loaded$counts
  celltype <- loaded$celltype

  if (!is.null(outfile)) {
    if (!is.null(celltype)) {
      celltype_aligned <- if (!is.null(names(celltype))) {
        celltype[colnames(counts)]
      } else if (length(celltype) == ncol(counts)) {
        celltype
      } else {
        warning("celltype length != ncol(counts); saving NA labels.")
        rep(NA, ncol(counts))
      }
      lab_file <- file.path(dirname(outfile), "celltype.txt")
      write.table(data.frame(cell = colnames(counts),
                             celltype = as.integer(factor(celltype_aligned)),
                             stringsAsFactors = FALSE),
                  file = lab_file, sep = "\t",
                  quote = FALSE, row.names = FALSE, col.names = TRUE)
      message("Saved cell labels to: ", lab_file)
    } else {
      message("No celltype found; skipping label export.")
    }

    if (isTRUE(save_counts_csv)) {
      csv_file <- if (grepl("\\.csv$", outfile, ignore.case = TRUE)) {
        outfile
      } else {
        paste0(outfile, ".counts.csv")
      }
      message("Writing counts CSV (dense) to: ", csv_file)
      write.csv(as.matrix(counts), file = csv_file, quote = FALSE)
    }
  }

  invisible(list(counts = counts, celltype = celltype))
}


# =========================================================================== #
# 3. The two datasets that do not arrive as an .h5
# =========================================================================== #

prep_seurat_object <- function(seurat_obj,
                               outfile_norm,
                               nfeatures = 3000,
                               assay = "RNA",
                               drop_zero_sum_genes = TRUE,
                               do_minmax = TRUE,
                               outfile_celltype = NULL,
                               verbose = FALSE) {
  # For a Seurat object carrying its annotation in Idents(), as the CD4 T cell data
  # (GSE310947) is published. Run UpdateSeuratObject() first if it is an old object.
  stopifnot(inherits(seurat_obj, "Seurat"))

  res <- scale_hvg_minmax(seurat_obj, assay = assay, nfeatures = nfeatures,
                          scale_features = "hvg",
                          drop_zero_sum_genes = drop_zero_sum_genes,
                          do_minmax = do_minmax, verbose = verbose)
  labels <- as.integer(as.character(Seurat::Idents(res$so)))
  paths <- write_dmvae_input(res$norm, labels, outfile_norm, outfile_celltype)

  invisible(list(so = res$so, hvgs = res$hvgs, norm = res$norm,
                 celltype = labels, files = paths))
}


prep_pbmc <- function(data_dir,
                      out_dir,
                      nfeatures = 4000,
                      n_sample = 2500,
                      seed = 123) {
  # Human PBMC (10x PBMC 3k). No published per-cell annotation, so reference labels
  # come from the Seurat tutorial's clustering and marker naming, collapsed to four
  # types after dropping DC and Platelet.
  suppressPackageStartupMessages(library(dplyr))

  pbmc.data <- Seurat::Read10X(data.dir = data_dir)
  pbmc <- Seurat::CreateSeuratObject(counts = pbmc.data, project = "pbmc3k",
                                     min.cells = 3, min.features = 200)
  pbmc[["percent.mt"]] <- Seurat::PercentageFeatureSet(pbmc, pattern = "^MT-")
  pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

  # PCA and the neighbour graph below read scale.data, so scale every gene.
  res <- scale_hvg_minmax(pbmc, nfeatures = nfeatures, scale_features = "all",
                          drop_zero_sum_genes = FALSE, do_minmax = FALSE)
  pbmc <- res$so

  pbmc <- Seurat::RunPCA(pbmc, features = Seurat::VariableFeatures(object = pbmc))
  pbmc <- Seurat::FindNeighbors(pbmc, dims = 1:10)
  pbmc <- Seurat::FindClusters(pbmc, resolution = 0.5)
  pbmc <- Seurat::RunUMAP(pbmc, dims = 1:10)

  new.cluster.ids <- c("Naive CD4 T", "CD14+ Mono", "Memory CD4 T", "B", "CD8 T",
                       "FCGR3A+ Mono", "NK", "DC", "Platelet")
  names(new.cluster.ids) <- levels(pbmc)
  pbmc <- Seurat::RenameIdents(pbmc, new.cluster.ids)
  pbmc <- subset(pbmc, idents = c("DC", "Platelet"), invert = TRUE)

  idents <- as.character(Seurat::Idents(pbmc))
  collapsed <- ifelse(idents == "B", "B",
               ifelse(idents == "NK", "NK",
               ifelse(idents %in% c("FCGR3A+ Mono", "CD14+ Mono"), "Mono", "CD4/CD8")))
  labels <- as.numeric(as.factor(collapsed))

  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  # Counts for the competing methods, plus the scDAC layout.
  count_hvg <- t(as.matrix(
    Seurat::GetAssayData(pbmc, assay = "RNA", slot = "counts")[Seurat::VariableFeatures(pbmc), ]
  ))
  write_scdac_layout(count_hvg, labels, out_dir)
  write.csv(as.matrix(Seurat::GetAssayData(pbmc, assay = "RNA", slot = "counts")),
            file.path(out_dir, "counts.csv"), quote = FALSE)

  # Model input; n_sample = NULL keeps all cells.
  X <- t(as.matrix(Seurat::GetAssayData(pbmc, assay = "RNA", slot = "scale.data")))
  keep <- colSums(X) != 0
  X <- X[, keep, drop = FALSE]

  if (!is.null(n_sample) && n_sample < nrow(X)) {
    set.seed(seed)
    idx <- sample(seq_len(nrow(X)), n_sample)
    X <- X[idx, , drop = FALSE]
    labels <- labels[idx]
  }

  norm <- as.data.frame(apply(X, 2, minmax_01))
  write_dmvae_input(norm, labels, file.path(out_dir, "data_norm.txt"))

  invisible(list(so = pbmc, norm = norm, celltype = labels))
}


# =========================================================================== #
# 4. Driver
# =========================================================================== #

preprocess_benchmark <- function(datasets = BENCHMARK_DATASETS, data_root = DATA_ROOT) {
  for (fname in datasets) {
    dataset_dir <- file.path(data_root, fname)
    h5_path <- file.path(dataset_dir, "data.h5")
    message("\n==== ", fname, " ====")
    tryCatch({
      prep_seurat_from_h5(
        h5_path = h5_path,
        fname   = fname,
        outfile = file.path(dataset_dir, "data_norm.txt")
      )
      extract_counts_from_h5(
        h5_path = h5_path,
        fname   = fname,
        outfile = file.path(dataset_dir, "counts.csv"),
        save_counts_csv = TRUE
      )
    }, error = function(e) {
      message("FAILED: ", fname, " | ", conditionMessage(e))
    })
  }
}


if (sys.nframe() == 0L && !interactive()) {
  requested <- commandArgs(trailingOnly = TRUE)
  preprocess_benchmark(if (length(requested)) requested else BENCHMARK_DATASETS)
}
