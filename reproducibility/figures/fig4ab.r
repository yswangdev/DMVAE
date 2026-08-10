#!/usr/bin/env Rscript
#
# Figure 4 (a, b) -- runs destiny's diffusion pseudotime on z_mean, then draws
# (a) the UMAP coloured by collection time point and (b) violin/box plots of the
# pseudotime rank per time point with Wilcoxon brackets.
#
# Panels (c) and (d) are in fig4cd.py.
#
# Usage:
#   Rscript fig4ab.r
#   Z_MEAN=/path/z_mean.txt LABEL_PATH=/path/labels.txt Rscript fig4ab.r
#
# Optional:
#   FIG_OUT_DIR  ROOT_LABEL=3  DM_N_EIGS=20  DM_K=50  DM_SIGMA=local
#   DPT_W_WIDTH=0.2  CLUSTER_POINT_SIZE=0.25
#

suppressPackageStartupMessages({
  if (!requireNamespace("destiny", quietly = TRUE)) {
    stop("Install destiny: BiocManager::install('destiny')", call. = FALSE)
  }
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Install ggplot2: install.packages('ggplot2')", call. = FALSE)
  }
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("Install uwot: install.packages('uwot')", call. = FALSE)
  }
  library(destiny)
  library(ggplot2)
  library(uwot)
})

theme_fig <- function() {
  theme_classic(base_size = 20) +
    theme(
      panel.grid = element_blank(),
      axis.text = element_text(size = 18),
      axis.title = element_text(size = 21),
      legend.text = element_text(size = 18),
      legend.title = element_text(size = 19),
      plot.title = element_text(size = 22)
    )
}

p_to_stars <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.001) return("***")
  if (p < 0.01) return("**")
  if (p < 0.05) return("*")
  "ns"
}

Z_MEAN <- Sys.getenv(
  "Z_MEAN",
  "/Volumes/SSD/MCW/Research/Aim 1/DMVAE/Klein/z_mean.txt"
)
LABEL_PATH <- Sys.getenv(
  "LABEL_PATH",
  "/Volumes/SSD/MCW/Research/Aim 1/Data/mouse_e/data_celltype.txt"
)
# Local paths by default; override FIG_OUT_DIR and the inputs to run on the cluster.
OUT_DIR <- Sys.getenv("FIG_OUT_DIR",
                      "/Volumes/SSD/MCW/Research/Aim 1/Documents/Paper_draft/papers/")
ROOT_LABEL <- Sys.getenv("ROOT_LABEL", "")
DM_N_EIGS <- as.integer(Sys.getenv("DM_N_EIGS", "20"))
DM_K <- as.integer(Sys.getenv("DM_K", "50"))
DM_SIGMA <- Sys.getenv("DM_SIGMA", "local")
DPT_W_WIDTH <- as.numeric(Sys.getenv("DPT_W_WIDTH", "0.2"))
CLUSTER_POINT_SIZE <- as.numeric(Sys.getenv("CLUSTER_POINT_SIZE", "0.25"))
PLOT_DPI <- 300L

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

message("Reading: ", Z_MEAN)
z <- as.matrix(utils::read.table(Z_MEAN, header = FALSE, comment.char = ""))
message("Reading: ", LABEL_PATH)
lab <- utils::read.table(LABEL_PATH, header = FALSE, comment.char = "")[, 1]

if (nrow(z) != length(lab)) {
  stop("Row count mismatch: z_mean rows=", nrow(z), " labels=", length(lab), call. = FALSE)
}

n_cells <- nrow(z)
cell_ids <- paste0("cell_", seq_len(n_cells))
rownames(z) <- cell_ids

root_lab <- if (nzchar(ROOT_LABEL)) as.numeric(ROOT_LABEL) else min(as.numeric(lab))
root_idx <- which(as.numeric(lab) == root_lab)[1]
if (is.na(root_idx)) stop("No cells found for ROOT_LABEL=", root_lab, call. = FALSE)
message("Root label: ", root_lab, " | root cell index: ", root_idx)

z_scaled <- scale(z)
sigma_arg <- if (DM_SIGMA == "local") "local" else as.numeric(DM_SIGMA)

message("DiffusionMap params: n_eigs=", DM_N_EIGS, " k=", DM_K, " sigma=", DM_SIGMA)
dm <- DiffusionMap(z_scaled, n_eigs = DM_N_EIGS, k = DM_K, sigma = sigma_arg)

message("Running DPT with w_width=", DPT_W_WIDTH)
dpt_raw <- as.numeric(DPT(dm, tips = root_idx, w_width = DPT_W_WIDTH)$dpt)

set.seed(42)
um <- uwot::umap(z, n_neighbors = 30, min_dist = 0.3, metric = "euclidean")

dpt_rank <- rank(dpt_raw, ties.method = "average", na.last = "keep")

tp_map <- c(`1` = "d0", `2` = "d2", `3` = "d4", `4` = "d7")
ct <- as.integer(lab)
tp_lbl <- unname(tp_map[as.character(ct)])
tp_lbl[is.na(tp_lbl)] <- as.character(ct[is.na(tp_lbl)])

df <- data.frame(
  UMAP1 = um[, 1],
  UMAP2 = um[, 2],
  cluster = factor(tp_lbl, levels = c("d0", "d2", "d4", "d7"))
)

p_umap <- ggplot(df, aes(UMAP1, UMAP2, color = cluster)) +
  geom_point(size = CLUSTER_POINT_SIZE) +
  scale_color_brewer(palette = "Set1", name = "Clusters") +
  labs(x = NULL, y = NULL) +
  guides(color = guide_legend(override.aes = list(size = 10))) +
  theme_fig() +
  theme(
    legend.key.size = grid::unit(13, "mm"),
    legend.text = element_text(size = 19),
    legend.title = element_text(size = 20),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank()
  )
out_umap <- file.path(OUT_DIR, "umap_colored_by_cluster.png")
ggsave(out_umap, p_umap, width = 6.5, height = 5, dpi = PLOT_DPI)
message("Wrote: ", out_umap)

df_v <- data.frame(
  dpt = dpt_rank,
  dpt_raw = dpt_raw,
  timepoint = factor(tp_lbl, levels = c("d0", "d2", "d4", "d7"))
)

p_v <- ggplot(df_v, aes(timepoint, dpt, fill = timepoint)) +
  geom_violin(scale = "width", trim = FALSE, alpha = 0.9, color = "black", linewidth = 0.35, show.legend = FALSE) +
  geom_boxplot(width = 0.12, outlier.size = 0.4, alpha = 1, color = "black", linewidth = 0.35, show.legend = FALSE) +
  scale_fill_brewer(palette = "Set1", drop = FALSE) +
  theme_fig() +
  xlab("Time points") +
  ylab("Diffusion pseudotime rank")

levels_present <- levels(droplevels(df_v$timepoint))
target_pairs <- list(c("d0", "d2"), c("d2", "d4"), c("d4", "d7"))
pair_list <- Filter(function(pr) all(pr %in% levels_present), target_pairs)

raw_pvals <- vapply(pair_list, function(pr) {
  x <- df_v$dpt_raw[df_v$timepoint == pr[1]]
  y <- df_v$dpt_raw[df_v$timepoint == pr[2]]
  stats::wilcox.test(x, y, exact = FALSE)$p.value
}, numeric(1))
pvals_adj <- stats::p.adjust(raw_pvals, method = "bonferroni")

y_top <- max(df_v$dpt, na.rm = TRUE)
y_span <- max(1, diff(range(df_v$dpt, na.rm = TRUE)))
step <- 0.10 * y_span
base <- y_top + 0.12 * y_span

for (k in seq_along(pair_list)) {
  x1 <- which(levels(df_v$timepoint) == pair_list[[k]][1])
  x2 <- which(levels(df_v$timepoint) == pair_list[[k]][2])
  yk <- base + (k - 1) * step
  p_lab <- p_to_stars(pvals_adj[k])
  p_v <- p_v +
    annotate("segment", x = x1, xend = x2, y = yk, yend = yk, linewidth = 0.4) +
    annotate("segment", x = x1, xend = x1, y = yk - 0.03 * y_span, yend = yk, linewidth = 0.4) +
    annotate("segment", x = x2, xend = x2, y = yk - 0.03 * y_span, yend = yk, linewidth = 0.4) +
    annotate("text", x = (x1 + x2) / 2, y = yk + 0.04 * y_span, label = p_lab, size = 7, fontface = "bold")
}

p_v <- p_v + coord_cartesian(ylim = c(min(df_v$dpt, na.rm = TRUE), base + length(pair_list) * step + 0.08 * y_span))
out_v <- file.path(OUT_DIR, "dpt_violin_boxplot_by_timepoint.png")
ggsave(out_v, p_v, width = 7, height = 4.5, dpi = PLOT_DPI)
message("Wrote: ", out_v)
message("Done.")
