library(data.table)

mix_dir    <- "/scratch/g/chlin/Yushu/Data/4_clus_mix"
scgnn_root <- "/scratch/g/chlin/Yushu/scGNN/Data/4_clus"

out_mix    <- "/scratch/g/chlin/Yushu/Data/4_clus_mix_downsample"
out_scgnn  <- "/scratch/g/chlin/Yushu/scGNN/Data/4_clus_downsample"

if (!dir.exists(out_mix))   dir.create(out_mix,   recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(out_scgnn)) dir.create(out_scgnn, recursive = TRUE, showWarnings = FALSE)

sims  <- 1:20
fracs <- c(0.8, 0.6, 0.4)
set.seed(20260211)

dir.create(out_mix,  recursive=TRUE, showWarnings=FALSE)
dir.create(out_scgnn,recursive=TRUE, showWarnings=FALSE)

for (frac in fracs) {
  
  tag <- sprintf("p%02d", round(frac * 100))   # p80 / p60 / p40
  
  # ---- one folder per percent ----
  out_mix_pct   <- file.path(out_mix, tag)
  out_scgnn_pct <- file.path(out_scgnn, tag)
  dir.create(out_mix_pct,   recursive = TRUE, showWarnings = FALSE)
  dir.create(out_scgnn_pct, recursive = TRUE, showWarnings = FALSE)
  
  for (sim in sims) {
    
    f_meta  <- file.path(mix_dir,  sprintf("simmeta_%d.txt", sim))
    f_data  <- file.path(mix_dir,  sprintf("simdata_%d.txt", sim))
    f_norm  <- file.path(mix_dir,  sprintf("simnorm_%d.txt", sim))
    f_count <- file.path(scgnn_root, sprintf("sim%d/simcounts_%d.csv", sim, sim))
    
    meta  <- fread(f_meta)
    data  <- fread(f_data)
    norm  <- fread(f_norm)
    count <- fread(f_count)
    
    n <- nrow(meta)
    stopifnot(nrow(data) == n, nrow(norm) == n, nrow(count) == n)
    
    k   <- floor(n * frac)
    idx <- sort(sample.int(n, k))  # 1-based row indices
    
    # ---- save index (no header/colname) ----
    fwrite(data.table(idx = idx),
           file.path(out_mix_pct, sprintf("sim%d_idx_1based.csv", sim)),
           row.names = FALSE, col.names = FALSE)
    
    # ---- write downsampled outputs (NO row names, NO col names) ----
    write.table(meta[idx, ],  file.path(out_mix_pct,  sprintf("simmeta_%d.txt", sim)),
           row.names = FALSE, col.names = FALSE)
    write.table(data[idx, ],  file.path(out_mix_pct,  sprintf("simdata_%d.txt", sim)),
           row.names = FALSE, col.names = FALSE)
    write.table(norm[idx, ],  file.path(out_mix_pct,  sprintf("simnorm_%d.txt", sim)),
           row.names = FALSE, col.names = FALSE)
    write.csv(count[idx, ],
              file.path(out_scgnn_pct, sprintf("simcounts_%d.csv", sim)),
              row.names = FALSE, col.names = FALSE, quote = FALSE)
    
    cat(sprintf("%s sim%d: %d/%d cells\n", tag, sim, k, n))
  }
}

# scdac
# ------------------------------------------------------------
# For each pct (p40/p60/p80) and sim1-20:
# 1) Select vec/*.csv by index file (1-based) where vec files are 0000.csv, 0001.csv, ...
#    => mapping: file_id = index - 1
# 2) Subset label.csv (cell starts at 0) by the SAME file_id set, keep same order
# 3) Create feat/feat_dims.csv with:
#    ,rna
#    1,2000
# ------------------------------------------------------------

pct_list <- c("p40", "p60", "p80")
sim_list <- 1:20

data_base  <- "/scratch/g/chlin/Yushu/scDAC/scDAC/data/4_clus"
index_base <- "/scratch/g/chlin/Yushu/Data/4_clus_mix_downsample"
out_base   <- "/scratch/g/chlin/Yushu/scDAC/scDAC/data/4clus_downsample"

pad_width  <- 4
file_ext   <- ".csv"
move_files <- FALSE  # TRUE = move, FALSE = copy
feat_dim   <- 2000L  # for feat_dims.csv

read_idx_firstcol <- function(path) {
  x <- read.csv(path, header = TRUE, stringsAsFactors = FALSE)
  idx <- x[[1]]
  idx <- as.integer(as.character(idx))
  idx <- idx[!is.na(idx)]
  idx
}

subset_and_write_label <- function(label_path, idx0, out_label_path) {
  lab <- read.csv(label_path, header = TRUE, stringsAsFactors = FALSE)
  
  if (!all(c("cell", "label") %in% names(lab))) {
    stop("label.csv must have columns: cell,label. Got: ", paste(names(lab), collapse = ", "))
  }
  
  lab$cell  <- as.integer(as.character(lab$cell))
  lab$label <- as.integer(as.character(lab$label))
  
  m <- match(idx0, lab$cell)  # keep same order as idx0
  if (anyNA(m)) {
    warning(sprintf("label.csv missing %d cells. Example: %s",
                    sum(is.na(m)), paste(head(idx0[is.na(m)], 5), collapse = ", ")))
  }
  
  sub_lab <- lab[m[!is.na(m)], , drop = FALSE]
  dir.create(dirname(out_label_path), showWarnings = FALSE, recursive = TRUE)
  write.csv(sub_lab, out_label_path, row.names = FALSE, quote = FALSE)
}

write_feat_dims <- function(feat_dir, dims = 2000L) {
  dir.create(feat_dir, showWarnings = FALSE, recursive = TRUE)
  df <- data.frame(rna = as.integer(dims))
  rownames(df) <- "1"
  # creates:
  # ,rna
  # 1,2000
  write.csv(df, file.path(feat_dir, "feat_dims.csv"), quote = FALSE)
}

for (pct in pct_list) {
  for (sim in sim_list) {
    
    # ---- CHANGED HERE: vec/rna ----
    data_dir  <- file.path(data_base,  paste0("sim", sim), "subset_0", "vec", "rna")
    index_csv <- file.path(index_base, pct, paste0("sim", sim, "_idx_1based.csv"))
    out_dir   <- file.path(out_base,   pct, paste0("sim", sim), "subset_0", "vec", "rna")
    
    # label + feat output (per sim)
    label_src <- file.path(data_base, paste0("sim", sim), "label.csv")
    label_out <- file.path(out_base,  pct, paste0("sim", sim), "label.csv")
    feat_dir  <- file.path(out_base,  pct, paste0("sim", sim), "feat")
    
    if (!dir.exists(data_dir))   { warning("Missing data_dir: ", data_dir); next }
    if (!file.exists(index_csv)) { warning("Missing index_csv: ", index_csv); next }
    if (!file.exists(label_src)) { warning("Missing label_src: ", label_src); next }
    
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    
    # index starts at 1, files start at 0000 -> subtract 1
    idx1 <- read_idx_firstcol(index_csv)
    idx0 <- idx1 - 1L
    idx0 <- idx0[idx0 >= 0L]
    
    # build 4-digit filenames like 0000.csv, 0001.csv, ...
    fn <- paste0(sprintf(paste0("%0", pad_width, "d"), idx0), file_ext)
    src_files <- file.path(data_dir, fn)
    
    ok <- file.exists(src_files)
    if (!all(ok)) {
      warning(sprintf("[%s sim%d] Missing %d vec files. Example: %s",
                      pct, sim, sum(!ok),
                      paste(head(fn[!ok], 5), collapse = ", ")))
    }
    src_files <- src_files[ok]
    if (length(src_files) == 0) {
      warning(sprintf("[%s sim%d] No vec files to copy after filtering.", pct, sim))
      next
    }
    
    dest_files <- file.path(out_dir, basename(src_files))
    
    if (move_files) {
      file.rename(src_files, dest_files)
    } else {
      file.copy(src_files, dest_files, overwrite = TRUE)
    }
    
    # ---- add label.csv subset ----
    subset_and_write_label(label_src, idx0, label_out)
    
    # ---- add feat_dims.csv creation ----
    write_feat_dims(feat_dir, dims = feat_dim)
    
    message(sprintf("[%s sim%d] vec=%d files -> %s | label -> %s | feat_dims -> %s",
                    pct, sim, length(src_files), out_dir, label_out,
                    file.path(feat_dir, "feat_dims.csv")))
  }
}


