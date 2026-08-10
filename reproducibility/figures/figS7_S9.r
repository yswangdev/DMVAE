#!/usr/bin/env Rscript
#
# Supplementary Figures S7, S8 and S9 -- the CD4 T1D dataset, at the width the
# main text could not carry:
#   S7  dot plot of the top 10 marker genes per cluster
#   S8  GSEA NES heatmaps, cluster-vs-rest
#   S9  GSEA NES heatmaps, WT-vs-KO
#
# Recomputes nothing: reads the marker and GSEA summary CSVs that fig5cf.r wrote,
# and sources fig5cf.r for its shared helpers. Run fig5cf.r first.
#
# Usage:
#   Rscript figS7_S9.r                 # both parts
#   PARTS=dot  Rscript figS7_S9.r      # dot plot only (needs the Seurat object)
#   PARTS=gsea Rscript figS7_S9.r      # heatmaps only (CSVs only; runs locally)
#
# ===========================================================================

PARTS <- trimws(strsplit(Sys.getenv("PARTS", "dot,gsea"), ",")[[1]])

# The guard stops fig5cf.r rendering a panel while we source it for its helpers.
FIG5_HELPERS_ONLY <- TRUE
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
                                                 value = TRUE)[1])), "fig5cf.r"))

result_path <- Sys.getenv(
  "RESULT_PATH",
  "/scratch/g/chlin/Yushu/results/dmvae/CD4/0224/1aelr_0_001_aep_30_lrnn_0_001_beta_0_3"
)
OUT_DIR <- Sys.getenv("SUPP_OUT_DIR",
                      file.path(result_path, "supplement"))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

sig_cutoff <- 0.05


# ===========================================================================
# DOT -- top 10 markers per cluster
# ===========================================================================
if ("dot" %in% PARTS) {

suppressPackageStartupMessages({
  library(Seurat); library(jsonlite); library(dplyr); library(ggplot2)
})

data_path <- Sys.getenv(
  "DATA_PATH",
  "/scratch/g/chlin/Yushu/Data/Example_bio_data/CD4_with_Treg.RData"
)
top_n_marker_plot <- 10
target_k <- c("4", "8", "13")

# Same ranking as fig5cf.r's safe_top_markers(), which lives inside a section
# that does not run when this file sources it.
top_markers <- function(markers_df, top_n) {
  fc_col <- if ("avg_log2FC" %in% colnames(markers_df)) "avg_log2FC" else "avg_logFC"
  markers_df %>%
    group_by(cluster) %>%
    {
      if ("p_val_adj_recalc" %in% colnames(.)) {
        arrange(., p_val_adj_recalc, desc(.data[[fc_col]]), .by_group = TRUE)
      } else {
        arrange(., desc(.data[[fc_col]]), .by_group = TRUE)
      }
    } %>%
    slice_head(n = top_n) %>%
    pull(gene) %>%
    unique()
}

load(data_path)
seu <- Filter(function(x) inherits(x, "Seurat"), mget(ls()))[[1]]

assign_all <- jsonlite::fromJSON(file.path(result_path, "assignments_all_k.json"),
                                 simplifyVector = TRUE)

for (k in target_k) {
  markers_csv <- file.path(result_path, "seurat_DE", "pct_0.2", paste0("k_", k),
                           paste0("markers_cluster_k", k, ".csv"))
  if (!file.exists(markers_csv)) {
    message("missing (run fig5cf.r first): ", markers_csv)
    next
  }

  labels <- as.integer(assign_all[[k]])
  present <- sort(unique(labels))
  remap <- setNames(seq_along(present) - 1L, as.character(present))
  labels <- unname(remap[as.character(labels)])
  # Labelled by the non-empty cluster count, as in fig5cf.r: k=8 renders as k=7.
  k_label <- length(present)

  cluster_col <- paste0("dmvae_k", k)
  seu[[cluster_col]] <- factor(labels, levels = as.character(seq_along(present) - 1L))

  markers_cluster <- utils::read.csv(markers_csv, stringsAsFactors = FALSE)
  genes <- top_markers(markers_cluster, top_n_marker_plot)
  genes <- genes[genes %in% rownames(seu)]
  if (length(genes) < 2) {
    message("not enough markers at k=", k)
    next
  }

  png_path <- file.path(OUT_DIR, paste0("S_dotplot_top", top_n_marker_plot,
                                        "_markers_k", k_label, ".png"))
  marker_dotplot(seu, genes, cluster_col, png_path)
  message("saved: ", png_path, "  (", length(genes), " genes)")
}

}  # end DOT


# ===========================================================================
# GSEA -- top 5 pathways per cluster, redrawn from the saved summary CSVs
# ===========================================================================
if ("gsea" %in% PARTS) {

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr)
  library(ComplexHeatmap); library(circlize); library(grid); library(ragg)
  library(stringr)
})

top_per_clust <- 5
dbs <- c("H", "C2", "C5_BP", "C7")
heatmap_orientation <- "vertical"
pct_tag <- "pct_0.2"

gsea_roots <- list(
  ClusterVsRest = "gsea_cluster_identity_by_k_human_pathway",
  WT_vs_KO      = "gsea_WTKO_by_k_human_pathway"
)

gsea_csv <- function(tag, k, k_tag, db) {
  file.path(result_path, paste0(gsea_roots[[tag]], "_", heatmap_orientation),
            pct_tag, paste0("k_", k), db,
            sprintf("GSEA_%s_%s_summary_%s.csv", db, tag, k_tag))
}

# One entry per heatmap; `clusters` restricts the columns.
specs <- list(
  list(tag = "ClusterVsRest", k = "4",  ktag = "k4",        clusters = NULL),
  list(tag = "ClusterVsRest", k = "13", ktag = "k13_treg",  clusters = c("2", "3", "7", "11")),
  list(tag = "WT_vs_KO",      k = "4",  ktag = "k4",        clusters = NULL)
)

for (sp in specs) {
  for (db in dbs) {
    f <- gsea_csv(sp$tag, sp$k, sp$ktag, db)
    if (!file.exists(f)) { message("missing: ", f); next }

    df <- readr::read_csv(f, show_col_types = FALSE) %>%
      mutate(cluster = as.character(cluster))
    if (!is.null(sp$clusters)) df <- df %>% filter(cluster %in% sp$clusters)
    if (!nrow(df)) { message("no rows after cluster filter: ", f); next }

    positive_only <- sp$tag == "ClusterVsRest"
    top <- select_pathways(df, top_per_clust, sig_cutoff,
                           positive_only = positive_only)

    png_path <- file.path(OUT_DIR, sp$tag,
                          sprintf("S_GSEA_%s_%s_heatmap_%s.png", db, sp$tag, sp$ktag))
    ok <- draw_heatmap(df, top, png_path, orientation = heatmap_orientation,
                       sig = sig_cutoff, col_order = sp$clusters)
    message(if (isTRUE(ok)) "saved: " else "skipped: ", png_path,
            "  (", length(top), " pathways)")
  }
}

}  # end GSEA

message("Done.")
