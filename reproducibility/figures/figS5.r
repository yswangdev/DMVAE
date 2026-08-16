#!/usr/bin/env Rscript
#
# Supplementary Figure S5 -- p(c = cluster_i | k=4) panels: the DMVAE soft cluster
# posterior drawn on the UMAP of the original expression data, styled to match
# fig4ab.r panel (a).
#
# One facet per predicted cluster at a chosen k, every cell coloured by its
# posterior probability of belonging to that cluster.
#
# Uses fig4ab.r's embedding: same z_mean file, same UMAP settings
# (n_neighbors=30, min_dist=0.3, euclidean) and same seed.
#
# Usage:
#   Rscript figS5.r
#   K=4 Rscript figS5.r
#   UMAP_INPUT=/path/data_norm.txt Rscript figS5.r   # embed the raw data instead
#
# Optional:
#   PCZ  DATA_PATH  LABEL_PATH  FIG_OUT_DIR  A_MIN  DROP_EMPTY=1  RELABEL=1
#   POINT_SIZE=0.25  UMAP_NN=30  UMAP_MIN_DIST=0.3  SEED=42
#

suppressPackageStartupMessages({
  for (p in c("ggplot2", "uwot")) {
    if (!requireNamespace(p, quietly = TRUE)) {
      stop("Install ", p, ": install.packages('", p, "')", call. = FALSE)
    }
  }
  library(ggplot2)
  library(uwot)
})

# Match fig4a exactly so the panels sit alongside it.
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

# --- minimal .npy reader ---------------------------------------------------
# Little-endian float32/float64, C order.
read_npy <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  magic <- readBin(con, "raw", 6L)
  if (!identical(magic[2:6], charToRaw("NUMPY"))) {
    stop("Not a .npy file: ", path, call. = FALSE)
  }
  ver <- as.integer(readBin(con, "raw", 2L))
  hlen <- if (ver[1] == 1L) {
    as.integer(readBin(con, "integer", 1L, size = 2L, signed = FALSE, endian = "little"))
  } else {
    as.integer(readBin(con, "integer", 1L, size = 4L, endian = "little"))
  }
  header <- rawToChar(readBin(con, "raw", hlen))

  descr <- sub(".*'descr'\\s*:\\s*'([^']+)'.*", "\\1", header)
  fortran <- grepl("'fortran_order'\\s*:\\s*True", header)
  shp_txt <- sub(".*'shape'\\s*:\\s*\\(([^)]*)\\).*", "\\1", header)
  shape <- as.integer(strsplit(gsub("\\s", "", shp_txt), ",")[[1]])
  shape <- shape[!is.na(shape)]

  size <- switch(sub("^[<>|=]", "", descr),
                 f4 = 4L, f8 = 8L,
                 stop("Unsupported dtype '", descr, "' in ", path, call. = FALSE))
  n <- prod(shape)
  x <- readBin(con, "double", n = n, size = size, endian = "little")

  if (length(shape) == 1L) return(x)
  if (fortran) {
    array(x, dim = shape)
  } else {
    # numpy C order fills the last axis fastest; R fills the first. Build with
    # reversed dims, then permute back.
    aperm(array(x, dim = rev(shape)), length(shape):1)
  }
}

# --- inputs ------------------------------------------------------------------
DIRECTORY <- Sys.getenv("DMVAE_DIRECTORY", ".")
RUN_DIR <- Sys.getenv("DMVAE_RUN_DIR", file.path(DIRECTORY, "results"))
PCZ <- Sys.getenv("PCZ", file.path(RUN_DIR, "p_c_z_best.npy"))
DATA_PATH <- Sys.getenv("DATA_PATH",
                        file.path(DIRECTORY, "Data", "mouse_e", "data_norm.txt"))
LABEL_PATH <- Sys.getenv("LABEL_PATH",
                         file.path(DIRECTORY, "Data", "mouse_e", "data_celltype.txt"))
# Embed the same z_mean fig4a embeds, with fig4a's UMAP settings and seed, so the
# point cloud here is identical to fig4a's and the two panels can be read together.
UMAP_INPUT <- Sys.getenv(
  "UMAP_INPUT",
  file.path(DIRECTORY, "DMVAE", "Klein", "z_mean.txt")
)
OUT_DIR <- Sys.getenv("FIG_OUT_DIR",
                      file.path(DIRECTORY, "results", "dmvae", "figures"))
K <- as.integer(Sys.getenv("K", "6"))
# k printed in the panel titles; defaults to K. Set it to report the realised
# (non-empty) cluster count instead. Changes the label only.
LABEL_K <- as.integer(Sys.getenv("LABEL_K", "4"))
A_MIN <- Sys.getenv("A_MIN", "")
DROP_EMPTY <- Sys.getenv("DROP_EMPTY", "1") == "1"
RELABEL <- Sys.getenv("RELABEL", "1") == "1"
POINT_SIZE <- as.numeric(Sys.getenv("POINT_SIZE", "0.25"))
UMAP_NN <- as.integer(Sys.getenv("UMAP_NN", "30"))
UMAP_MIN_DIST <- as.numeric(Sys.getenv("UMAP_MIN_DIST", "0.3"))
SEED <- as.integer(Sys.getenv("SEED", "42"))
PLOT_DPI <- 300L

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

message("Reading posterior: ", PCZ)
pcz <- read_npy(PCZ)
if (length(dim(pcz)) != 3L) {
  stop("Expected a 3-D posterior (cell, k, component), got dim ",
       paste(dim(pcz), collapse = "x"), call. = FALSE)
}
n_cells <- dim(pcz)[1]; n_k <- dim(pcz)[2]; b <- dim(pcz)[3]
# The k axis holds one entry per k in [a, b] and the component axis is padded to
# b, so a follows exactly from the shape.
a <- if (nzchar(A_MIN)) as.integer(A_MIN) else b - n_k + 1L
message("  shape ", paste(dim(pcz), collapse = " x "), "  ->  k range [", a, ", ", b, "]")
if (K < a || K > b) stop("K=", K, " outside the searched range [", a, ", ", b, "]", call. = FALSE)
k_idx <- K - a + 1L   # R is 1-based
message("  plotting k=", K, " (slice ", k_idx, ")")

P <- pcz[, k_idx, seq_len(K), drop = TRUE]   # cells x K

# The unnormalised density is floored at 1e-10 before normalising: if every
# component underflows, the posterior collapses to a flat 1/b and the padded
# components hold (b-K)/b.
if (K < b) {
  pad <- rowSums(pcz[, k_idx, (K + 1L):b, drop = FALSE])
  degen <- (b - K) / b
  frac <- mean(abs(pad - degen) < 1e-3)
  message(sprintf("  padding mass: mean %.5f (degenerate would be %.4f); %.1f%% of cells at it",
                  mean(pad), degen, 100 * frac))
  if (frac > 0.01) {
    warning("Some cells have a numerically floored, uniform posterior. Their ",
            "apparent uncertainty is an artifact, not a lineage continuum.",
            call. = FALSE)
  }
}

message("Reading labels: ", LABEL_PATH)
lab <- utils::read.table(LABEL_PATH, header = FALSE, comment.char = "")[, 1]
if (length(lab) != n_cells) {
  stop("Label count ", length(lab), " != posterior cells ", n_cells, call. = FALSE)
}

# --- embedding ---------------------------------------------------------------
embed_src <- if (nzchar(UMAP_INPUT)) UMAP_INPUT else DATA_PATH
message("Reading embedding input: ", embed_src)
X <- as.matrix(utils::read.table(embed_src, header = FALSE, comment.char = ""))
X[is.na(X)] <- 0
if (nrow(X) != n_cells) {
  stop("Embedding input has ", nrow(X), " rows but the posterior has ", n_cells,
       " cells -- these are not the same dataset.", call. = FALSE)
}
message("  running UMAP on ", nrow(X), " x ", ncol(X),
        " (n_neighbors=", UMAP_NN, ", min_dist=", UMAP_MIN_DIST, ", seed=", SEED, ")")
set.seed(SEED)
um <- uwot::umap(X, n_neighbors = UMAP_NN, min_dist = UMAP_MIN_DIST, metric = "euclidean")

# --- which components to draw ------------------------------------------------
hard <- max.col(P, ties.method = "first")          # 1..K
occupied <- sort(unique(hard))
if (length(occupied) < K) {
  message("  ", length(occupied), " of ", K, " components are occupied: ",
          paste(occupied - 1L, collapse = ", "),
          "  (empty: ", paste(setdiff(seq_len(K), occupied) - 1L, collapse = ", "), ")")
}
keep <- if (DROP_EMPTY) occupied else seq_len(K)
# Dropping unused components leaves gaps in the numbering; those gaps reflect
# which components the model happened to use, not anything about the clusters.
disp <- if (RELABEL) seq_along(keep) else keep - 1L
message("  panels: ", paste(sprintf("c=%d -> %d", keep - 1L, disp), collapse = ", "))

# --- long frame, low probabilities first so high ones draw on top -------------
df <- do.call(rbind, lapply(seq_along(keep), function(j) {
  data.frame(UMAP1 = um[, 1], UMAP2 = um[, 2], p = P[, keep[j]],
             cluster = factor(sprintf("p(c = %d | k = %d)", disp[j], LABEL_K),
                              levels = sprintf("p(c = %d | k = %d)", disp, LABEL_K)))
}))
df <- df[order(df$cluster, df$p), ]

p_fig <- ggplot(df, aes(UMAP1, UMAP2, color = p)) +
  geom_point(size = POINT_SIZE) +
  # "free" scales put the axis lines on each panel. Every facet holds the same cells,
  # so the ranges come out identical.
  facet_wrap(~cluster, nrow = 2, scales = "free") +
  # Sequential ramp, monotonic in lightness; limits fixed at 0..1.
  scale_color_gradientn(
    colours = c("#E6E6E6", "#EFB8A6", "#DD7B62", "#C1392F", "#8E1015"),
    limits = c(0, 1), name = "p(c | k)"
  ) +
  labs(x = NULL, y = NULL) +
  theme_fig() +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(size = 21),
    legend.key.height = grid::unit(18, "mm"),
    axis.line = element_line(colour = "black", linewidth = 0.5),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    panel.spacing = grid::unit(8, "mm")
  )

# Name by the slice plotted; note in the filename when the printed label differs.
base <- sprintf("figS5_posterior_k%d", LABEL_K)
out_png <- file.path(OUT_DIR, paste0(base, ".png"))
out_pdf <- file.path(OUT_DIR, paste0(base, ".pdf"))
ggsave(out_png, p_fig, width = 11, height = 9, dpi = PLOT_DPI)
ggsave(out_pdf, p_fig, width = 11, height = 9)
message("Wrote: ", out_png)
message("Wrote: ", out_pdf)

# --- numeric form of the figure ----------------------------------------------
tp_map <- c(`1` = "d0", `2` = "d2", `3` = "d4", `4` = "d7")
tp <- unname(tp_map[as.character(as.integer(lab))])
tp[is.na(tp)] <- as.character(as.integer(lab))[is.na(tp)]
message("\nmean p(c=i | k=", K, ") by true group:")
tab <- t(sapply(unique(tp), function(g) round(colMeans(P[tp == g, , drop = FALSE]), 3)))
colnames(tab) <- paste0("c=", seq_len(K) - 1L)
print(tab)
message("\nmax posterior per cell, by group (median):")
print(round(tapply(apply(P, 1, max), tp, median), 3))
message("Done.")
