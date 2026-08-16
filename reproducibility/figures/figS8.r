#!/usr/bin/env Rscript
#
# Supplementary Figure S8 -- Seurat UMAPs of the Type 1 diabetes CD4 T-cell
# dataset at resolutions 0.2, 0.8, and 1.0, plus the curated annotation.
#
# Usage:
#   Rscript figS8.r
#   DATA_PATH=/path/CD4_with_Treg.RData FIGURE_OUTPUT_ROOT=/path/out Rscript figS8.r
#

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(patchwork)
})

DIRECTORY <- Sys.getenv("DMVAE_DIRECTORY", ".")
DATA_PATH <- Sys.getenv(
  "DATA_PATH",
  file.path(DIRECTORY, "Data", "Example_bio_data", "CD4_with_Treg.RData")
)
OUT_DIR <- Sys.getenv(
  "FIGURE_OUTPUT_ROOT",
  file.path(DIRECTORY, "results", "dmvae", "figures")
)
ANNOTATION_COLUMN <- Sys.getenv("ANNOTATION_COLUMN", "merged_annotation")
RESOLUTIONS <- as.numeric(strsplit(Sys.getenv("SEURAT_RESOLUTIONS", "0.2,0.8,1.0"),
                                  ",")[[1]])
PLOT_DPI <- as.integer(Sys.getenv("PLOT_DPI", "300"))

if (!file.exists(DATA_PATH)) stop("Data file not found: ", DATA_PATH, call. = FALSE)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

load(DATA_PATH)
objects <- Filter(function(x) inherits(x, "Seurat"), mget(ls()))
if (!length(objects)) stop("No Seurat object found in ", DATA_PATH, call. = FALSE)
seu <- objects[[1]]

if (!("umap" %in% names(seu@reductions))) {
  if (!("pca" %in% names(seu@reductions))) {
    DefaultAssay(seu) <- if ("SCT" %in% Assays(seu)) "SCT" else DefaultAssay(seu)
    seu <- RunPCA(seu, npcs = 30, verbose = FALSE)
  }
  seu <- RunUMAP(seu, dims = 1:min(30, ncol(Embeddings(seu, "pca"))),
                 seed.use = 42, verbose = FALSE)
}

graph_name <- paste0(DefaultAssay(seu), "_snn")
if (!(graph_name %in% names(seu@graphs))) {
  if (!("pca" %in% names(seu@reductions))) {
    seu <- RunPCA(seu, npcs = 30, verbose = FALSE)
  }
  seu <- FindNeighbors(seu, dims = 1:min(30, ncol(Embeddings(seu, "pca"))),
                       verbose = FALSE)
  graph_name <- paste0(DefaultAssay(seu), "_snn")
}

resolution_columns <- character(length(RESOLUTIONS))
for (i in seq_along(RESOLUTIONS)) {
  value <- RESOLUTIONS[i]
  seu <- FindClusters(seu, graph.name = graph_name, resolution = value,
                      random.seed = 42, verbose = FALSE)
  column <- paste0("figS8_resolution_", format(value, trim = TRUE, scientific = FALSE))
  seu[[column]] <- factor(Idents(seu))
  resolution_columns[i] <- column
}

if (!(ANNOTATION_COLUMN %in% colnames(seu[[]]))) {
  stop("Annotation column '", ANNOTATION_COLUMN, "' not found. Available columns: ",
       paste(colnames(seu[[]]), collapse = ", "), call. = FALSE)
}

annotation_names <- c(
  "0" = "0-Memory", "1" = "1-Naive",
  "3" = "3-Il21+ early effector/Tfh-like", "4" = "4-Il21+ Th1-like",
  "5" = "5-Effector memory", "6" = "6-Proliferating",
  "7" = "7-Acinar contamination", "8" = "8-Naive",
  "treg_0" = "Treg_0", "treg_1" = "Treg_1", "treg_2" = "Treg_2",
  "treg_3" = "Treg_3", "treg_4" = "Treg_4", "treg_5" = "Treg_5"
)
raw_annotation <- as.character(seu@meta.data[[ANNOTATION_COLUMN]])
display_annotation <- unname(annotation_names[raw_annotation])
display_annotation[is.na(display_annotation)] <- raw_annotation[is.na(display_annotation)]
seu$figS8_annotation <- factor(display_annotation)

theme_s8 <- theme_classic(base_size = 13) +
  theme(axis.title = element_blank(), axis.text = element_blank(),
        axis.ticks = element_blank(), legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold"))

panels <- lapply(seq_along(RESOLUTIONS), function(i) {
  DimPlot(seu, reduction = "umap", group.by = resolution_columns[i],
          pt.size = 0.25, raster = TRUE, label = TRUE, repel = TRUE,
          label.size = 3.5) +
    ggtitle(paste0("Resolution ", format(RESOLUTIONS[i], trim = TRUE))) +
    theme_s8 + NoLegend()
})
panels[[length(panels) + 1L]] <-
  DimPlot(seu, reduction = "umap", group.by = "figS8_annotation",
          pt.size = 0.25, raster = TRUE, label = TRUE, repel = TRUE,
          label.size = 3.0) +
  ggtitle("CD4 with Treg annotation") + theme_s8 + NoLegend()

figure <- wrap_plots(panels, ncol = 2)
png_path <- file.path(OUT_DIR, "figS8_seurat_resolution_umap.png")
pdf_path <- file.path(OUT_DIR, "figS8_seurat_resolution_umap.pdf")
ggsave(png_path, figure, width = 13, height = 10, dpi = PLOT_DPI)
ggsave(pdf_path, figure, width = 13, height = 10)
message("Saved: ", png_path)
message("Saved: ", pdf_path)
