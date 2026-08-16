#!/usr/bin/env Rscript
#
# Figure 5 (c-f) -- islet-infiltrating CD4 T cells:
#   (c) stacked WT/KO composition per cluster at k = 4, 7, 13
#   (d) dot plot of the top marker genes per cluster
#   (e) GSEA NES heatmaps, cluster-vs-rest, at k = 4 and k = 13 (Treg)
#   (f) GSEA NES heatmap, WT-vs-KO, at k = 4
#
# Panels (a) and (b) are in fig5ab.py. These are the main-text selections, 5 marker
# genes and 2 pathways per cluster; figS9_S11.r draws the wider supplementary
# versions and reuses the shared helpers below.
#
# Usage:
#   Rscript fig5cf.r                    # both sections
#   PANELS=cd Rscript fig5cf.r          # panels c and d
#   PANELS=ef Rscript fig5cf.r          # panels e and f (full GSEA)
#
# ===========================================================================

# figS9_S11.r sources this file for the shared helpers below and sets
# FIG5_HELPERS_ONLY first, so sourcing defines them without rendering a panel.
PANELS <- if (exists("FIG5_HELPERS_ONLY")) {
  character(0)
} else {
  trimws(strsplit(Sys.getenv("PANELS", "cd,ef"), ",")[[1]])
}

# Every figure image goes here; the CSV/RDS analysis outputs stay beside the run
# so figS9_S11.r can redraw from them without recomputing the GSEA.
DIRECTORY <- Sys.getenv("DMVAE_DIRECTORY", ".")
FIG_OUT_DIR <- Sys.getenv(
  "FIG_OUT_DIR", file.path(DIRECTORY, "results", "dmvae", "figures"))
dir.create(FIG_OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# One NES range for every pathway heatmap, so a colour means the same in (e) and
# (f). Just above the widest panel's |NES| so the ramp is fully used.
NES_LEGEND_LIMIT <- 3

# Ticks on integers, endpoints included: -3..3 at 3, -4,-2,0,2,4 at 4.
nes_ticks <- function(m) {
  step <- if (m == round(m)) (if (m %% 2 == 0) m / 2 else 1) else m / 2
  s <- seq(0, m, by = step)
  sort(unique(c(-s, s)))
}


# ===========================================================================
# SHARED HELPERS -- used by the sections below and by figS9_S11.r
# ===========================================================================

# Panel (c) colours mirror fig5ab.py panel (a) rule for rule: annotation order,
# palette index, Hungarian match, split shading.
NUM_NAMES <- c(
  "0" = "0-Memory",
  "1" = "1-Naive",
  "3" = "3-Il21+early effect/T fh-like",
  "4" = "4-Il21+Th1",
  "5" = "5-Effector memory",
  "6" = "6-Proliferating",
  "7" = "7-Acinar contamination",
  "8" = "8-Naive"
)

# Shades for the 2nd, 3rd, ... cluster matching one annotation.
SPLIT_SHADES <- c(-0.55, 0.55, -0.75, 0.75)

# Per-cell annotation label -> display name (treg_k -> Treg_k).
annotation_display <- function(x) {
  x <- as.character(x)
  out <- x
  is_treg <- startsWith(tolower(x), "treg")
  if (any(is_treg)) {
    out[is_treg] <- paste0(
      "Treg_", vapply(strsplit(x[is_treg], "_"), function(p) p[2], character(1)))
  }
  hit <- !is_treg & x %in% names(NUM_NAMES)
  out[hit] <- unname(NUM_NAMES[x[hit]])
  out
}

# Display names in legend order: numeric clusters ascending, then Treg subclusters.
annotation_order <- function(x) {
  x <- as.character(x)
  is_treg <- startsWith(tolower(x), "treg")
  nums <- sort(unique(suppressWarnings(as.integer(x[!is_treg]))))
  nums <- nums[!is.na(nums)]
  tregs <- sort(unique(suppressWarnings(as.integer(
    vapply(strsplit(x[is_treg], "_"), function(p) p[2], character(1))))))
  tregs <- tregs[!is.na(tregs)]
  num_lab <- vapply(as.character(nums), function(n)
    if (n %in% names(NUM_NAMES)) unname(NUM_NAMES[[n]]) else n, character(1))
  c(unname(num_lab), if (length(tregs)) paste0("Treg_", tregs) else character(0))
}

# factor > 0 toward white, < 0 toward black.
shade_colour <- function(hex, factor) {
  v <- as.numeric(grDevices::col2rgb(hex)[, 1]) / 255
  v <- if (factor >= 0) v + (1 - v) * factor else v * (1 + factor)
  grDevices::rgb(v[1], v[2], v[3])
}

# Colour each cluster like the annotation it best matches (maximum-overlap
# assignment); clusters sharing one annotation are shaded apart, largest pure.
matched_cluster_colors <- function(assign_k, disp, ann_order, palette) {
  if (!requireNamespace("clue", quietly = TRUE)) {
    stop("Install clue for the cluster/annotation matching: install.packages('clue')",
         call. = FALSE)
  }
  clusters <- sort(unique(assign_k))
  w <- matrix(0, nrow = length(clusters), ncol = length(ann_order))
  for (ci in seq_along(clusters)) {
    m <- assign_k == clusters[ci]
    for (oi in seq_along(ann_order)) w[ci, oi] <- sum(m & disp == ann_order[oi])
  }

  # solve_LSAP needs ncol >= nrow; pad so a surplus cluster goes unmatched.
  w_pad <- w
  if (ncol(w_pad) < nrow(w_pad)) {
    w_pad <- cbind(w_pad, matrix(0, nrow(w_pad), nrow(w_pad) - ncol(w_pad)))
  }
  assigned <- as.integer(clue::solve_LSAP(w_pad, maximum = TRUE))

  mapping <- integer(length(clusters))
  for (ci in seq_along(clusters)) {
    oi <- assigned[ci]
    if (oi > length(ann_order) || w[ci, oi] == 0) {
      oi <- if (sum(w[ci, ]) > 0) which.max(w[ci, ]) else 0L
    }
    mapping[ci] <- oi
  }

  sizes <- vapply(clusters, function(cl) sum(assign_k == cl), numeric(1))
  out <- setNames(rep("#cccccc", length(clusters)), as.character(clusters))
  for (oi in unique(mapping)) {
    idx <- which(mapping == oi)
    idx <- idx[order(-sizes[idx])]
    base <- if (oi >= 1) palette[((oi - 1) %% length(palette)) + 1] else "#cccccc"
    for (r in seq_along(idx)) {
      f <- if (r == 1) 0 else SPLIT_SHADES[((r - 2) %% length(SPLIT_SHADES)) + 1]
      out[idx[r]] <- shade_colour(base, f)
    }
  }
  out
}

# Normalise for matching; the manuscript writes names without the GSE prefix.
norm_pathway <- function(x) {
  x <- toupper(as.character(x))
  x <- iconv(x, to = "ASCII//TRANSLIT")
  x <- gsub("_", " ", x)
  x <- gsub("^GSE[0-9]+ ", "", x)
  x <- gsub("[^A-Z0-9 ]", "", x)
  trimws(gsub(" +", " ", x))
}

# Plus plural collapse, so "CD4 TCELLS 2H" meets MSigDB's "CD4_TCELL_2H".
vague_tokens <- function(x) {
  x <- norm_pathway(x)
  x <- gsub("\\b([A-Z0-9]{4,})S\\b", "\\1", x)
  strsplit(trimws(gsub(" +", " ", x)), " ")
}

# Tokens of `want` in `have`, same order, extra words allowed.
in_order <- function(want_tok, have_tok) {
  i <- 1L
  for (t in have_tok) {
    if (i > length(want_tok)) break
    if (t == want_tok[i]) i <- i + 1L
  }
  i > length(want_tok)
}

# Matched per cluster: C7 has several accessions per name.
pinned_in_cluster <- function(dc, want) {
  if (!length(want)) return(character(0))
  have <- vague_tokens(dc$Description)
  out <- character(0)
  for (w in vague_tokens(want)) {
    # Exact wins when present, so inserted words never cost precision.
    idx <- which(vapply(have, function(h) identical(h, w), logical(1)))
    if (!length(idx)) {
      idx <- which(vapply(have, function(h) in_order(w, h), logical(1)))
    }
    if (!length(idx)) next
    out <- c(out, as.character(dc$Description[idx[which.max(abs(dc$NES[idx]))]]))
  }
  unique(out)
}

# Rows for one heatmap. Per cluster: a manuscript-named pathway if it has one,
# else that cluster's top `n`. `pinned` holds NAMES, matched per cluster.
select_pathways <- function(df, n, sig, positive_only = FALSE,
                            pinned = character(0)) {
  sig_df <- df %>% filter(p.adjust < sig)
  if (!nrow(sig_df)) return(character(0))

  # Numeric cluster order, matching the heatmap columns.
  cls <- unique(as.character(sig_df$cluster))
  cl_num <- suppressWarnings(as.integer(cls))
  cls <- if (!any(is.na(cl_num))) cls[order(cl_num)] else sort(cls)

  out <- character(0)
  for (cl in cls) {
    dc <- sig_df %>% filter(as.character(cluster) == cl)

    # Kept whatever its sign: the text discusses several as downregulated.
    hit <- pinned_in_cluster(dc, pinned)
    if (length(hit) > 0) {
      out <- c(out, hit)
      next
    }

    # Otherwise cluster-vs-rest takes positive enrichments only.
    if (positive_only) dc <- dc %>% filter(NES > 0)
    if (!nrow(dc)) next
    dc <- if (positive_only) dc %>% arrange(p.adjust, desc(NES))
          else dc %>% arrange(desc(abs(NES)))
    out <- c(out, head(as.character(dc$Description), n))
  }
  unique(out)
}

# NES heatmap; only cells passing `sig` are coloured.
draw_heatmap <- function(summary_df, top, png_path, orientation = "vertical",
                         sig = 0.05, col_order = NULL,
                         nes_limit = NES_LEGEND_LIMIT,
                         path_font = 22, cluster_font = 26) {
  if (!length(top)) return(invisible(FALSE))
  if (length(nes_limit) != 1 || !is.finite(nes_limit) || nes_limit <= 0) {
    stop("`nes_limit` must be one positive finite number.", call. = FALSE)
  }

  clusters <- if (is.null(col_order)) {
    sort(unique(as.character(summary_df$cluster)))
  } else {
    intersect(col_order, unique(as.character(summary_df$cluster)))
  }
  cells <- summary_df %>%
    filter(Description %in% top, p.adjust < sig) %>%
    group_by(Description, cluster) %>%
    slice_max(abs(NES), n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    select(Description, cluster, NES)
  full <- expand_grid(Description = top, cluster = clusters) %>%
    left_join(cells, by = c("Description", "cluster"))

  nes <- full %>%
    pivot_wider(names_from = cluster, values_from = NES)
  rn <- nes$Description; nes$Description <- NULL
  nes <- as.matrix(nes); rownames(nes) <- rn
  if (is.null(col_order)) {
    nes <- nes[, order(suppressWarnings(as.integer(colnames(nes)))), drop = FALSE]
  } else {
    nes <- nes[, intersect(col_order, colnames(nes)), drop = FALSE]
  }
  nes[is.na(nes)] <- 0

  wrap_width    <- 30      # wrap pathway names early to keep the panel compact
  # pathway-name and cluster-name font sizes come from the caller; panel (f)
  # is rendered larger than the panel (e) heatmaps.
  cell_width_in <- 0.60    # narrow columns without compressing the rows
  cell_height_in <- 0.80   # heatmap row height, inches
  legend_in     <- 3.0     # keep pathway names clear of the colour legend
  # One fixed range across panels; colorRamp2 clamps beyond it.
  m_abs <- nes_limit
  if (any(abs(nes) > m_abs)) {
    message("    ", sum(abs(nes) > m_abs), " cell(s) clipped at |NES| = ", m_abs)
  }
  legend_param <- list(title_gp = gpar(fontsize = 22, fontface = "bold"),
                       labels_gp = gpar(fontsize = 20),
                       title_gap = unit(5, "mm"),
                       at = nes_ticks(m_abs),
                       legend_height = unit(2.2, "in"),
                       grid_width = unit(0.45, "in"))
  col_fun <- colorRamp2(c(-m_abs, 0, m_abs), c("navy", "white", "firebrick3"))

  wrap_lab <- function(x) str_wrap(str_replace_all(x, "_", " "), wrap_width)
  char_in  <- 0.62 * path_font / 72
  longest_label_in <- function(x) {
    lines <- unlist(strsplit(wrap_lab(x), "\n"))
    max(nchar(lines)) * char_in + 0.3
  }

  if (orientation == "horizontal") {
    nes <- t(nes)
    label_inches <- longest_label_in(colnames(nes))
    ht <- Heatmap(
      nes, name = "NES", col = col_fun,
      width  = unit(cell_width_in * ncol(nes), "in"),
      height = unit(cell_height_in * nrow(nes), "in"),
      cluster_rows = FALSE, cluster_columns = FALSE,
      rect_gp = gpar(col = "grey92", lwd = 0.4),
      column_labels = wrap_lab(colnames(nes)),
      column_names_rot = 90, column_names_side = "bottom",
      column_names_gp = gpar(fontsize = path_font),
      column_names_max_height = unit(label_inches, "in"),
      row_names_gp = gpar(fontsize = cluster_font),
      heatmap_legend_param = legend_param
    )
    w_in <- cell_width_in * ncol(nes) + 2.0 + legend_in
    h_in <- cell_height_in * nrow(nes) + label_inches + 1.0
  } else {
    label_inches <- longest_label_in(rownames(nes))
    ht <- Heatmap(
      nes, name = "NES", col = col_fun,
      width  = unit(cell_width_in * ncol(nes), "in"),
      height = unit(cell_height_in * nrow(nes), "in"),
      cluster_rows = FALSE, cluster_columns = FALSE,
      rect_gp = gpar(col = "grey92", lwd = 0.4),
      row_labels = wrap_lab(rownames(nes)),
      row_names_side = "right",
      row_names_gp = gpar(fontsize = path_font),
      row_names_max_width = unit(label_inches, "in"),
      # ComplexHeatmap justifies rather than centres these at rot = 0.
      column_names_rot = 0, column_names_side = "bottom",
      column_names_centered = TRUE,
      column_names_gp = gpar(fontsize = cluster_font),
      heatmap_legend_param = legend_param
    )
    w_in <- cell_width_in * ncol(nes) + label_inches + legend_in - 0.5
    h_in <- cell_height_in * nrow(nes) + 1.0
  }

  dir.create(dirname(png_path), recursive = TRUE, showWarnings = FALSE)
  agg_png(png_path, width = w_in, height = h_in, units = "in", res = 300,
          background = "white")
  draw(ht, heatmap_legend_side = "right")
  dev.off()
  invisible(TRUE)
}

# Marker dot plot for one clustering.
marker_dotplot <- function(seu, genes, cluster_col, png_path) {
  p_dot <- DotPlot(object = seu, features = genes, group.by = cluster_col) +
    RotatedAxis() +
    # Shorter than Seurat's "Percent Expressed" / "Average Expression".
    labs(size = "% Expressed", colour = "Avg. Expr.", fill = "Avg. Expr.") +
    theme(
      axis.text.x  = element_text(size = 18),
      axis.text.y  = element_text(size = 20),
      axis.title   = element_text(size = 22),
      legend.text  = element_text(size = 18),
      legend.title = element_text(size = 19),
      legend.key.size = grid::unit(9, "mm"),
      # RotatedAxis() labels run down-and-left, so the first gene overhangs.
      plot.margin = grid::unit(c(0.1, 0.2, 0.35, 0.9), "in")
    )
  # Width tracks the gene count; the extra inch matches the left margin.
  ggsave(png_path, p_dot, width = max(12, DOTPLOT_IN_PER_GENE * length(genes)) + 1.0,
         height = 7.5, dpi = 300)
  invisible(png_path)
}


# ===========================================================================
# SECTION 1 -- panels (c) and (d)
# ===========================================================================
if ("cd" %in% PANELS) {
###################### plots and DE for all k's ######################
dataset = "CD4_with_Treg" # or Treg, venus

# ------------------- paths -------------------
if(dataset == "CD4"){
  data_path   <- file.path(DIRECTORY, "Data", "Example_bio_data", "CD4.RData")
  result_path <- file.path(DIRECTORY, "results", "dmvae", "CD4", "0219",
                           "1aelr_0_001_aep_40_lrnn_0_001_beta_0_3")
}else if(dataset == "Treg"){
  data_path   <- file.path(DIRECTORY, "Data", "Example_bio_data", "Treg.RData")
  result_path <- file.path(DIRECTORY, "results", "dmvae", "Treg", "0225",
                           "1aelr_0_001_aep_30_lrnn_5e-04_beta_0_3")
}else if(dataset == "venus"){
  data_path   <- file.path(DIRECTORY, "Data", "Example_bio_data", "venus.RData")
  result_path <- file.path(DIRECTORY, "results", "dmvae", "venus", "0219",
                           "1aelr_0_001_aep_30_lrnn_1e-04_beta_0_3")
}else if(dataset == "CD4_with_Treg"){
  data_path   <- file.path(DIRECTORY, "Data", "Example_bio_data",
                           "CD4_with_Treg.RData")
  result_path <- file.path(DIRECTORY, "results", "dmvae", "CD4", "0224",
                           "1aelr_0_001_aep_30_lrnn_0_001_beta_0_3")
}else{
  stop("Unknown dataset: ", dataset)
}
data_path <- Sys.getenv("DATA_PATH", data_path)
result_path <- Sys.getenv("RESULT_PATH", result_path)

all_k_assignment <- file.path(result_path, "assignments_all_k.json")
base_out_dir <- file.path(result_path, "seurat_DE")

# ------------------- libs -------------------
suppressPackageStartupMessages({
  library(Seurat)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

# ------------------- config -------------------
genotype_col <- "WT_KO"
run_wtko <- TRUE
# Horizontal room per gene, so rotated names sit apart.
DOTPLOT_IN_PER_GENE <- 0.62

# Panel (c) bar stacking, top to bottom, matching the rendered Sankey.
# Keyed by the REQUESTED k, so "8" is the panel labelled k=7.
BAR_STACK_ORDER <- list(
  "4"  = c(0, 1, 2, 3),
  "8"  = c(3, 5, 6, 2, 0, 4, 1),
  "13" = c(4, 10, 12, 0, 2, 11, 3, 7, 6, 5, 8, 9, 1)
)

# Panel (c) canvas; the extra width is the cluster key.
BAR_WIDTH_IN  <- 4.6
BAR_HEIGHT_IN <- 6

# Markers per cluster in the dot plot; figS9_S11.r uses 10.
top_n_marker_plot <- 5

# Gene symbols the panel must always include, whatever their rank.
main_text_genes <- character(0)
min_cells_cluster <- 10

# Directory holding cluster_colors_k*.csv written by `python fig5ab.py --panel a`.
CLUSTER_COLOR_DIR <- Sys.getenv("CLUSTER_COLOR_DIR",
                                file.path(result_path, "umap_plots"))

# Fallback palette when the exported colour map is absent.
DISTINCT_COLORS <- c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                     "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
                     "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
                     "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3")

# Minimum expression proportion for a marker.
min_pct_values <- c(0.20)

# transferred annotation column in the merged object
merged_annotation_col <- "Treg_annotation"

# ------------------- helper -------------------
safe_top_markers <- function(markers_df, top_n = 5) {
  if (is.null(markers_df) || nrow(markers_df) == 0) return(character(0))
  
  fc_col <- if ("avg_log2FC" %in% colnames(markers_df)) {
    "avg_log2FC"
  } else if ("avg_logFC" %in% colnames(markers_df)) {
    "avg_logFC"
  } else {
    stop("Cannot find avg_log2FC or avg_logFC column in marker table.")
  }
  
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

filter_marker_pct <- function(df, min_pct = 0.1) {
  if (is.null(df) || nrow(df) == 0) return(df)
  if (!all(c("pct.1", "pct.2") %in% colnames(df))) {
    warning("pct.1/pct.2 not found, returning unfiltered marker table.")
    return(df)
  }
  df %>%
    filter(pmax(pct.1, pct.2) >= min_pct)
}

recompute_fdr_and_rank <- function(df, group_col = NULL) {
  if (is.null(df) || nrow(df) == 0) return(df)
  if (!("p_val" %in% colnames(df))) {
    warning("p_val not found, skip FDR recalculation/ranking.")
    return(df)
  }
  
  fc_col <- if ("avg_log2FC" %in% colnames(df)) {
    "avg_log2FC"
  } else if ("avg_logFC" %in% colnames(df)) {
    "avg_logFC"
  } else {
    NULL
  }
  
  if (!is.null(group_col) && group_col %in% colnames(df)) {
    out <- df %>%
      group_by(.data[[group_col]]) %>%
      mutate(p_val_adj_recalc = p.adjust(p_val, method = "BH")) %>%
      {
        if (!is.null(fc_col)) {
          arrange(., p_val_adj_recalc, desc(abs(.data[[fc_col]])), p_val, .by_group = TRUE)
        } else {
          arrange(., p_val_adj_recalc, p_val, .by_group = TRUE)
        }
      } %>%
      mutate(rank_fdr_recalc = row_number()) %>%
      ungroup()
  } else {
    out <- df %>%
      mutate(p_val_adj_recalc = p.adjust(p_val, method = "BH")) %>%
      {
        if (!is.null(fc_col)) {
          arrange(., p_val_adj_recalc, desc(abs(.data[[fc_col]])), p_val)
        } else {
          arrange(., p_val_adj_recalc, p_val)
        }
      } %>%
      mutate(rank_fdr_recalc = row_number())
  }
  
  out
}

# ------------------- load Seurat object -------------------
load(data_path)
seurat_candidates <- Filter(function(x) inherits(x, "Seurat"), mget(ls()))
if (length(seurat_candidates) == 0) stop("No Seurat object found in the loaded .RData.")
if (length(seurat_candidates) > 1) {
  message("Multiple Seurat objects found: ", paste(names(seurat_candidates), collapse = ", "))
  message("Using the first one. If that's wrong, set `seu <- <correct_object>` manually.")
}
seu <- seurat_candidates[[1]]

if (!(genotype_col %in% colnames(seu@meta.data))) {
  stop("meta.data$", genotype_col, " not found.")
}
seu[[genotype_col]] <- factor(as.character(seu@meta.data[[genotype_col]]))
if (!all(c("WT", "KO") %in% levels(seu[[genotype_col]][, 1]))) {
  stop("WT and KO are not both present in meta.data$", genotype_col)
}


# ------------------- read assignments_all_k.json -------------------
if (!file.exists(all_k_assignment)) {
  stop("assignments_all_k.json not found at: ", all_k_assignment)
}

assign_all <- jsonlite::fromJSON(all_k_assignment, simplifyVector = TRUE)
k_keys <- names(assign_all)
if (is.null(k_keys) || length(k_keys) == 0) stop("No k entries found in assignments_all_k.json")
target_k <- c("4", "8", "13")
missing_k <- setdiff(target_k, k_keys)
if (length(missing_k) > 0) {
  stop("Requested k not found in assignments_all_k.json: ", paste(missing_k, collapse = ", "))
}
k_keys <- target_k

n_cells <- ncol(seu)

# ------------------- run DE per k -------------------
for (min_pct_marker in min_pct_values) {
  pct_tag <- formatC(min_pct_marker, format = "f", digits = 1)
  out_dir <- file.path(base_out_dir, paste0("pct_", pct_tag))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  message("\n########################################")
  message("Running min_pct_marker = ", min_pct_marker, " -> ", out_dir)
  
  for (k in k_keys) {
    message("\n==============================")
    message("Processing k = ", k)
    
    # -------- create folders for this k --------
    k_dir <- file.path(out_dir, paste0("k_", k))
    dir.create(k_dir, showWarnings = FALSE, recursive = TRUE)
    
    dotheat_dir <- file.path(k_dir, "de_plots")
    dir.create(dotheat_dir, showWarnings = FALSE, recursive = TRUE)

    supp_dir <- file.path(k_dir, "supplement")
    dir.create(supp_dir, showWarnings = FALSE, recursive = TRUE)

    labels <- as.integer(assign_all[[k]])
    if (length(labels) != n_cells) {
      stop("Length mismatch for k=", k, ": JSON has ", length(labels),
           " labels but Seurat has ", n_cells,
           ". Need DMVAE cell order mapping to align correctly.")
    }

    # Drop empty clusters and renumber 0..n-1, as fig5ab.py does.
    present <- sort(unique(labels))
    remap <- setNames(seq_along(present) - 1L, as.character(present))
    labels <- unname(remap[as.character(labels)])
    cl_levels_k <- as.character(seq_along(present) - 1L)
    # Figures are labelled by NON-EMPTY cluster count (k=8 renders as k=7);
    # the analysis outputs keep the requested k for figS9_S11.r.
    k_label <- length(present)
    message("k=", k, " cluster remap (orig -> new): ",
            paste(sprintf("%s->%s", names(remap), remap), collapse = ", "),
            if (k_label != as.integer(k)) paste0("  [figures labelled k=", k_label, "]") else "")

    cluster_col <- paste0("dmvae_k", k)
    seu[[cluster_col]] <- factor(labels, levels = cl_levels_k)
    Idents(seu) <- cluster_col
    
    # -------- (A) cluster markers --------
    markers_cluster_raw <- FindAllMarkers(
      object = seu,
      test.use = "wilcox",
      logfc.threshold = 0.0,
      only.pos = TRUE
    )
    
    markers_cluster <- filter_marker_pct(markers_cluster_raw, min_pct = min_pct_marker)
    markers_cluster <- recompute_fdr_and_rank(markers_cluster, group_col = "cluster")
    
    outA_csv <- file.path(k_dir, paste0("markers_cluster_k", k, ".csv"))
    outA_rds <- file.path(k_dir, paste0("markers_cluster_k", k, ".rds"))
    outA_raw_csv <- file.path(k_dir, paste0("markers_cluster_k", k, "_raw.csv"))
    outA_raw_rds <- file.path(k_dir, paste0("markers_cluster_k", k, "_raw.rds"))
    
    write.csv(markers_cluster, outA_csv, row.names = FALSE)
    saveRDS(markers_cluster, outA_rds)
    write.csv(markers_cluster_raw, outA_raw_csv, row.names = FALSE)
    saveRDS(markers_cluster_raw, outA_raw_rds)
    
    message("Saved filtered cluster markers: ", outA_csv)
    message("Saved raw cluster markers: ", outA_raw_csv)
    
    # -------- (B) WT vs KO within each cluster --------
    cluster_ids <- levels(Idents(seu))
    markers_wtko_all <- list()
    
    for (cl in cluster_ids) {
      cells_cl <- WhichCells(seu, idents = cl)
      if (length(cells_cl) < min_cells_cluster) next
      
      seu_cl <- subset(seu, cells = cells_cl)
      g <- as.character(seu_cl@meta.data[[genotype_col]])
      if (!all(c("WT", "KO") %in% unique(g))) next
      
      mk_raw <- FindMarkers(
        object = seu_cl,
        group.by = genotype_col,
        ident.1 = "WT",
        ident.2 = "KO",
        test.use = "wilcox",
        logfc.threshold = 0.0
      )
      
      mk_raw$gene <- rownames(mk_raw)
      mk_raw$cluster <- cl
      mk_raw$contrast <- "WT_vs_KO"
      
      mk <- filter_marker_pct(mk_raw, min_pct = min_pct_marker)
      mk <- recompute_fdr_and_rank(mk)
      markers_wtko_all[[cl]] <- mk
    }
    
    if (length(markers_wtko_all) > 0) {
      markers_wtko_df <- bind_rows(markers_wtko_all)
      
      outB_csv <- file.path(k_dir, paste0("markers_WTvsKO_byCluster_k", k, ".csv"))
      outB_rds <- file.path(k_dir, paste0("markers_WTvsKO_byCluster_k", k, ".rds"))
      write.csv(markers_wtko_df, outB_csv, row.names = FALSE)
      saveRDS(markers_wtko_df, outB_rds)
      message("Saved filtered WT vs KO-by-cluster markers: ", outB_csv)
    } else {
      markers_wtko_df <- NULL
      message("No WT vs KO-by-cluster markers produced for k=", k)
    }
    
    # -------- (C) composition plot --------
    df <- seu@meta.data %>%
      transmute(
        cluster = as.factor(.data[[cluster_col]]),
        WT_KO   = as.factor(.data[[genotype_col]])
      ) %>%
      filter(!is.na(cluster), !is.na(WT_KO))
    
    cl_chars <- as.character(df$cluster)
    suppressWarnings(cl_nums <- as.integer(cl_chars))
    if (all(!is.na(cl_nums))) {
      cl_levels <- as.character(sort(unique(cl_nums)))
    } else {
      cl_levels <- unique(cl_chars)
    }
    df$cluster <- factor(cl_chars, levels = cl_levels)
    df$WT_KO <- factor(df$WT_KO, levels = c("WT", "KO"))
    # Colours per k, from panel (a)'s export or recomputed by the same rule --
    # never a positional palette, which would disagree with panel (a).
    color_csv <- file.path(CLUSTER_COLOR_DIR, paste0("cluster_colors_k", k, ".csv"))
    label_file <- Sys.getenv("CD4_LABEL_FILE",
                             file.path(result_path, "CD4_with_Treg_label.txt"))
    if (file.exists(color_csv)) {
      cc <- utils::read.csv(color_csv, colClasses = "character")
      cluster_cols <- setNames(cc$color, cc$cluster)
      message("k=", k, ": colours from ", basename(color_csv))
    } else if (file.exists(label_file)) {
      raw_lab <- trimws(gsub('"', "", readLines(label_file)))
      raw_lab <- raw_lab[nzchar(raw_lab)]
      stopifnot(length(raw_lab) == n_cells)
      cluster_cols <- matched_cluster_colors(
        labels, annotation_display(raw_lab), annotation_order(raw_lab),
        DISTINCT_COLORS)
      message("k=", k, ": colours matched to the annotation")
    } else {
      stop("no ", basename(color_csv), " in ", CLUSTER_COLOR_DIR,
           " and no ", basename(label_file), " in ", result_path, call. = FALSE)
    }

    # Stacking order is pinned here, not read from the CSV. geom_col stacks by
    # the factor levels, first on top, and both bars share the one factor.
    want <- as.character(BAR_STACK_ORDER[[as.character(k)]])
    if (setequal(want, cl_levels)) {
      cl_levels <- want
      message("k=", k, ": bars stacked ", paste(want, collapse = ","), " (top down)")
    } else if (length(want)) {
      warning("BAR_STACK_ORDER[['", k, "']] != clusters present; using id order",
              call. = FALSE, immediate. = TRUE)
    }
    df$cluster <- factor(as.character(df$cluster), levels = cl_levels)
    cluster_cols <- cluster_cols[cl_levels]
    stopifnot(!anyNA(cluster_cols))

    df_pct <- df %>%
      count(WT_KO, cluster, name = "n") %>%
      group_by(WT_KO) %>%
      mutate(pct = 100 * n / sum(n)) %>%
      arrange(desc(cluster), .by_group = TRUE) %>%
      mutate(
        ymin = cumsum(pct) - pct,
        ymax = cumsum(pct),
        ymid = (ymin + ymax) / 2
      ) %>%
      ungroup()
    
    df_bounds <- df_pct %>%
      select(WT_KO, cluster, ymin, ymax) %>%
      tidyr::pivot_wider(
        names_from = WT_KO,
        values_from = c(ymin, ymax)
      )
    
    bar_width <- 0.42
    bar_half  <- bar_width / 2
    edge_pad  <- 0.05
    # Centre-to-centre distance between the WT and KO bars.
    bar_gap   <- 0.55
    x_wt      <- 1
    x_ko      <- x_wt + bar_gap

    df_pct$x    <- ifelse(df_pct$WT_KO == "WT", x_wt, x_ko)
    df_bounds$x <- x_wt

    p_left <- ggplot(df_pct, aes(x = x, y = pct, fill = cluster)) +
      geom_col(width = bar_width, color = NA) +
      scale_fill_manual(values = cluster_cols) +
      scale_x_continuous(breaks = c(x_wt, x_ko), labels = c("WT", "KO"),
                         expand = expansion(add = bar_half + edge_pad)) +
      coord_cartesian(ylim = c(0, 100), clip = "off") +
      labs(x = NULL, y = NULL) +
      theme_classic() +
      theme(
        panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.title = element_blank(),
        axis.text.x = element_text(size = 34),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.line.x = element_line(color = "black", linewidth = 0.5),
        axis.line.y = element_line(color = "black", linewidth = 0.5),
        legend.position = "right",
        legend.title = element_text(size = 26),
        legend.text = element_text(size = 24),
        legend.key.size = grid::unit(9, "mm"),
        plot.margin = margin(5.5, 5.5, 5.5, 5.5)
      ) +
      guides(fill = guide_legend(title = "Cluster", ncol = 1)) +
      geom_segment(
        data = df_bounds,
        aes(x = x_wt + bar_half, xend = x_ko - bar_half, y = ymin_WT, yend = ymin_KO),
        inherit.aes = FALSE,
        linetype = "dashed",
        color = "grey60",
        linewidth = 0.9
      ) +
      geom_segment(
        data = df_bounds,
        aes(x = x_wt + bar_half, xend = x_ko - bar_half, y = ymax_WT, yend = ymax_KO),
        inherit.aes = FALSE,
        linetype = "dashed",
        color = "grey75",
        linewidth = 0.4
      )
    
    ggsave(
      file.path(FIG_OUT_DIR, paste0("fig5c_bar_WT_KO_k", k_label, ".png")),
      # Fixed height so the k=4/7/13 panels line up when assembled.
      p_left, width = BAR_WIDTH_IN, height = BAR_HEIGHT_IN, dpi = 300
    )
    message("Saved composition plot")
    
    
    # -------- (D) marker dot plot --------
    top_genes <- safe_top_markers(markers_cluster, top_n = top_n_marker_plot)
    extra_genes <- setdiff(main_text_genes, top_genes)
    top_genes <- c(top_genes, extra_genes)
    top_genes <- top_genes[top_genes %in% rownames(seu)]
    if (length(extra_genes) > 0) {
      message("Added ", length(intersect(extra_genes, rownames(seu))),
              " main-text gene(s) to the dot plot")
    }

    if (length(top_genes) > 1) {
      dot_png <- marker_dotplot(
        seu, top_genes, cluster_col,
        file.path(FIG_OUT_DIR, paste0("fig5d_dotplot_topMarkers_k", k_label, ".png")))
      message("Saved dot plot: ", dot_png)

    } else {
      message("Not enough top genes for the DotPlot at k=", k)
    }
  }
}

message("All Done")
}  # end section 1

# ===========================================================================
# SECTION 2 -- panels (e) and (f): DE, GSEA, and the NES heatmaps.
#   WT_vs_KO      -> k = 4                      -> panel (f)
#   ClusterVsRest -> k = 4, and k = 13 (Treg)   -> panel (e)
# ===========================================================================
if ("ef" %in% PANELS) {
###################### GSEA: WT/KO and cluster identity, for all k ######################

suppressPackageStartupMessages({
  library(Seurat); library(jsonlite); library(dplyr); library(tidyr)
  library(clusterProfiler); library(msigdbr); library(ComplexHeatmap)
  library(circlize); library(grid); library(ragg); library(stringr)
})

# ------------------- config -------------------
dataset <- "CD4_with_Treg"
paths <- list(
  CD4 = list(
    data = file.path(DIRECTORY, "Data", "Example_bio_data", "CD4.RData"),
    res  = file.path(DIRECTORY, "results", "dmvae", "CD4", "0219",
                     "1aelr_0_001_aep_40_lrnn_0_001_beta_0_3")
  ),
  CD4_with_Treg = list(
    data = file.path(DIRECTORY, "Data", "Example_bio_data", "CD4_with_Treg.RData"),
    res  = file.path(DIRECTORY, "results", "dmvae", "CD4", "0224",
                     "1aelr_0_001_aep_30_lrnn_0_001_beta_0_3")
  )
)
stopifnot(dataset %in% names(paths))
data_path   <- paths[[dataset]]$data
result_path <- paths[[dataset]]$res
data_path <- Sys.getenv("DATA_PATH", data_path)
result_path <- Sys.getenv("RESULT_PATH", result_path)

genotype_col  <- "WT_KO"
target_k      <- c("4", "8", "13")
min_pct_vals  <- c(0.20)
# Collection shown in the main text; the rest are supplementary (figS9_S11.r).
MAIN_TEXT_DB <- "C7"

# Pathways per cluster; figS9_S11.r uses 5.
top_per_clust <- 2

# Pathways each panel must include, spelled as the manuscript writes them; matching
# is vague (see vague_tokens / in_order) so the prose spellings find the MSigDB sets.
# Kept per panel so one panel cannot pin another's signatures.
MAIN_TEXT_PATHWAYS <- list(
  ClusterVsRest = list(
    "4" = c(
      "NAIVE VS KLRG1HIGH EFF CD8 TCELL UP",
      "TREG VS TCONV UP",
      "ACUTE VS CHRONIC LCMV PRIMARY INF CD8 TCELL UP"
    ),
    "13" = c(
      "NAIVE VS EFF CD8 TCELL DN",
      "KAECH NAIVE VS DAY8 EFF CD8 TCELL DN",
      "WT VS A2AR TREG DN"
    )
  ),
  WT_vs_KO = list(
    "4" = c(
      "UNTREATED VS IL2 TREATED CD8 TCELL DAY6 POST IMMUNIZATION UP",
      "UNTREATED VS IL12 TREATED ACT CD4 TCELLS 2H DN",
      "TREG VS TCONV UP"
    )
  )
)

# Names the manuscript cites for one panel; empty when it cites none.
panel_pathways <- function(contrast, k) {
  by_k <- MAIN_TEXT_PATHWAYS[[contrast]]
  if (is.null(by_k)) return(character(0))
  want <- by_k[[as.character(k)]]
  if (is.null(want)) character(0) else want
}

# Resolve each main-text name to at most one Description in `df`. norm_pathway()
# drops the GSE##### prefix, so accessions can collapse to the same name; keep the
# most strongly enriched significant one.
resolve_main_text <- function(df, sig, want) {
  if (!length(want)) return(character(0))
  d <- df %>% filter(p.adjust < sig)
  if (!nrow(d)) return(character(0))
  have <- vague_tokens(d$Description)
  out <- character(0)
  for (w in vague_tokens(want)) {
    idx <- which(vapply(have, function(h) in_order(w, h), logical(1)))
    if (!length(idx)) next
    out <- c(out, as.character(d$Description[idx[which.max(abs(d$NES[idx]))]]))
  }
  unique(out)
}
sig_cutoff    <- 0.05
min_cells     <- 10

k_by_contrast <- list(WT_vs_KO = c("4"), ClusterVsRest = c("4", "13"))

# Cluster subset per k; NULL keeps all. The k=13 Treg clusters are 2, 3, 7, 11.
cluster_sel <- list("4" = NULL, "13" = c("2", "3", "7", "11"))

heatmap_orientation <- "vertical"
stopifnot(heatmap_orientation %in% c("horizontal", "vertical"))

gsea_dbs <- list(
  H     = list(category = "H",  subcategory = NULL),
  C2    = list(category = "C2", subcategory = NULL),
  C5_BP = list(category = "C5", subcategory = "BP"),
  C7    = list(category = "C7", subcategory = NULL)
)

contrasts <- list(
  WT_vs_KO      = list(root = "gsea_WTKO_by_k_human_pathway",             tag = "WT_vs_KO"),
  ClusterVsRest = list(root = "gsea_cluster_identity_by_k_human_pathway", tag = "ClusterVsRest")
)

# ------------------- helpers -------------------
ranked_vec <- function(de) {
  de <- de[!is.na(de$p_val) & !is.na(de$avg_log2FC), , drop = FALSE]
  s <- -log10(pmax(de$p_val, 1e-300)) * sign(de$avg_log2FC)
  names(s) <- toupper(rownames(de))
  s <- s[is.finite(s) & names(s) != ""]
  s <- s[order(-s)]
  s[!duplicated(names(s))]
}

run_gsea <- function(genes, t2g) {
  if (length(genes) < 10) return(NULL)
  tryCatch(
    GSEA(genes, TERM2GENE = t2g, minGSSize = 10, maxGSSize = 500,
         pvalueCutoff = 1, pAdjustMethod = "BH", verbose = FALSE, seed = TRUE),
    error = function(e) { message("    GSEA failed: ", conditionMessage(e)); NULL }
  )
}

run_de <- function(seu, cl, contrast, gcol, min_pct) {
  de <- if (contrast == "WT_vs_KO") {
    seu_cl <- subset(seu, idents = cl)
    g <- as.character(seu_cl@meta.data[[gcol]])
    if (!all(c("WT", "KO") %in% unique(g))) return(NULL)
    FindMarkers(seu_cl, group.by = gcol, ident.1 = "WT", ident.2 = "KO",
                test.use = "wilcox", logfc.threshold = 0, densify = FALSE)
  } else {
    FindMarkers(seu, ident.1 = cl, test.use = "wilcox",
                logfc.threshold = 0, densify = FALSE)
  }
  de <- de[pmax(de$pct.1, de$pct.2) >= min_pct, , drop = FALSE]
  if (!nrow(de)) NULL else de
}

# ------------------- load data -------------------
load(data_path)
seu <- Filter(function(x) inherits(x, "Seurat"), mget(ls()))[[1]]
seu[[genotype_col]] <- factor(as.character(seu@meta.data[[genotype_col]]))
stopifnot(all(c("WT", "KO") %in% levels(seu[[genotype_col]][, 1])))

assign_all <- jsonlite::fromJSON(file.path(result_path, "assignments_all_k.json"))
stopifnot(all(target_k %in% names(assign_all)),
          ncol(seu) == length(assign_all[[target_k[1]]]))

t2g_list <- lapply(gsea_dbs, function(d) {
  msigdbr(species = "Homo sapiens", category = d$category, subcategory = d$subcategory) %>%
    mutate(gene_symbol = toupper(gene_symbol)) %>%
    select(gs_name, gene_symbol) %>%
    distinct()
})

# ------------------- main loop -------------------
for (mp in min_pct_vals) {
  pct_tag <- sprintf("pct_%.1f", mp)
  message("\n############## min_pct = ", mp, " ##############")
  
  for (ctr_name in names(contrasts)) {
    ctr <- contrasts[[ctr_name]]
    out_dir <- file.path(result_path,
                         paste0(ctr$root, "_", heatmap_orientation),
                         pct_tag)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    message("\n>>> contrast: ", ctr_name)
    
    ks <- k_by_contrast[[ctr_name]]
    
    for (k in ks) {
      message("== k = ", k)
      seu$dmvae <- factor(as.integer(assign_all[[k]]))
      Idents(seu) <- "dmvae"
      summaries <- setNames(vector("list", length(t2g_list)), names(t2g_list))
      
      sel <- if (ctr_name == "ClusterVsRest") cluster_sel[[k]] else NULL
      cl_levels <- levels(Idents(seu))
      if (!is.null(sel)) cl_levels <- intersect(cl_levels, sel)
      
      for (cl in cl_levels) {
        if (sum(Idents(seu) == cl) < min_cells) next
        message("  cluster ", cl)
        de <- run_de(seu, cl, ctr_name, genotype_col, mp)
        if (is.null(de)) next
        genes <- ranked_vec(de)
        if (length(genes) < 10) next
        
        for (db in names(t2g_list)) {
          res <- run_gsea(genes, t2g_list[[db]])
          if (is.null(res) || nrow(res) == 0) next
          df <- as.data.frame(res); df$cluster <- cl; df$k <- k; df$database <- db
          summaries[[db]] <- bind_rows(summaries[[db]], df)
        }
      }
      
      k_tag <- if (ctr_name == "ClusterVsRest" && k == "13") "k13_treg" else paste0("k", k)
      for (db in names(summaries)) {
        df <- summaries[[db]]
        if (is.null(df) || !nrow(df)) next
        db_dir <- file.path(out_dir, paste0("k_", k), db)
        dir.create(db_dir, recursive = TRUE, showWarnings = FALSE)
        base <- file.path(db_dir, sprintf("GSEA_%s_%s_summary_%s", db, ctr$tag, k_tag))
        write.csv(df, paste0(base, ".csv"), row.names = FALSE)
        saveRDS(df, paste0(base, ".rds"))
        
        # (e)/(f) are C7 only; the rest are saved for figS9_S11.r.
        if (db != MAIN_TEXT_DB) {
          message("    ", db, " ", k_tag, ": summary saved (supplementary)")
          next
        }

        want <- panel_pathways(ctr_name, k)
        top <- select_pathways(df, top_per_clust, sig_cutoff,
                               positive_only = (ctr_name == "ClusterVsRest"),
                               pinned = want)
        pinned <- resolve_main_text(df, sig_cutoff, want)   # for the report only
        missing_p <- setdiff(norm_pathway(want), norm_pathway(pinned))
        if (length(missing_p) > 0) {
          message("    main-text pathway(s) not significant here: ",
                  paste(missing_p, collapse = "; "))
        }
        message("    ", db, " ", k_tag, ": ", length(top), " pathway row(s), ",
                length(intersect(top, pinned)), " from the manuscript")
        top_df <- df %>% filter(Description %in% top)
        top_base <- file.path(db_dir, sprintf("GSEA_%s_%s_heatmap_pathways_%s", db, ctr$tag, k_tag))
        write.csv(top_df, paste0(top_base, ".csv"), row.names = FALSE)
        saveRDS(top_df, paste0(top_base, ".rds"))

        # Panel (f) sits alone on its row, so its axis type is larger than the
        # panel (e) heatmaps generated by the same function.
        bigger <- if (ctr_name == "WT_vs_KO") 8 else 0
        
        draw_heatmap(
          df, top,
          file.path(FIG_OUT_DIR,
                    sprintf("fig5ef_GSEA_%s_%s_heatmap_%s.png", db, ctr$tag, k_tag)),
          orientation = heatmap_orientation,
          sig = sig_cutoff,
          path_font = 22 + bigger,
          cluster_font = 26 + bigger
        )
      }
    }
  }
}

message("All done.")

}  # end section 2
