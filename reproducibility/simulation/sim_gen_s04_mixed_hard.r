# Scenario s04 -- Splatter: 9 equal clusters, weak DE, 9,000 cells, 20 replicates.
#
# 2,000 genes, de.prob = 0.20, de.facLoc = 0.30, de.facScale = 0.20,
# de.downProb = 0.50, experiment-level dropout (mid = 1, shape = -1).
# 500 HVGs selected with modelGeneVar.

library(doParallel)
library(splatter)
library(scran)
library(scater)

numCores <- detectCores() - 4
cl <- makeCluster(numCores)
registerDoParallel(cl)
nsim <- 20

sim_root <- "/Volumes/SSD/MCW/Research/Codes/Simulation_single_cell/dmvae_sim"
path <- paste0(file.path(sim_root, "s04_mixed_hard"), "/")
.sim_out_root <- Sys.getenv("SIM_OUT_ROOT", unset = "")
if (nzchar(.sim_out_root)) path <- paste0(file.path(.sim_out_root, basename(sub("/+$", "", path))), "/")
if (!dir.exists(path)) {
  dir.create(path, recursive = TRUE)
}

params <- newSplatParams(nGenes = 2000, batchCells = 9000)

process_simulation <- function(i) {
  sim <- splatSimulateGroups(
    params,
    nGenes = 2000,
    group.prob = rep(1 / 9, 9),
    de.prob = rep(0.2, 9),
    de.facLoc = 0.30,
    de.facScale = rep(0.20, 9),
    de.downProb = 0.5,
    verbose = FALSE,
    dropout.type = "experiment",
    dropout.mid = 1,
    dropout.shape = -1,
    seed = i + 8001
  )

  counts_data <- t(as.matrix(counts(sim)))

  sim <- logNormCounts(sim)
  sim <- runPCA(sim)
  data <- as.data.frame(as.matrix(t(logcounts(sim))))

  if (sum(apply(data, 2, function(x) sum(x) == 0)) > 0) {
    data <- data[, -which(apply(data, 2, function(x) sum(x) == 0))]
  }
  scale_data <- as.data.frame(scale(data))

  sim_meta <- sim@colData@listData$Group
  sim_meta <- as.data.frame(sim_meta)
  sim_meta$sim_meta <- as.numeric(sim_meta$sim_meta)

  write.table(sim_meta, paste0(path, "simmeta_", i, ".txt"), row.names = FALSE, col.names = FALSE)
  write.table(counts_data, paste0(path, "simcounts_", i, ".csv"), sep = ",", row.names = FALSE, col.names = FALSE)

  # Raw-count h5 (X = cells x genes, Y = labels) for scAce/ADClust/scVI (github interface)
  h5f <- paste0(path, "sim_", i, ".h5")
  if (file.exists(h5f)) file.remove(h5f)
  rhdf5::h5createFile(h5f)
  .Xc <- t(counts_data); storage.mode(.Xc) <- "integer"
  rhdf5::h5write(.Xc, h5f, "X")
  rhdf5::h5write(as.integer(sim_meta[[1]]), h5f, "Y")
  rhdf5::H5close()
  write.table(data, paste0(path, "simdata_", i, ".txt"), row.names = FALSE, col.names = FALSE)

  # HVG-500 selection (matches sim_gen_scRNAseq_4c.r)
  sim_1 <- computeSumFactors(sim, sizes = c(20, 40, 60, 80))
  var.out <- modelGeneVar(sim_1, method = "loess")
  hvgs <- getTopHVGs(var.out, n = 500)
  data_scale_selected <- scale_data[, names(scale_data) %in% hvgs]

  data_norm <- as.data.frame(lapply(data_scale_selected, function(x) {
    if (is.numeric(x)) (x - min(x)) / (max(x) - min(x)) else x
  }))
  write.table(data_norm, paste0(path, "simnorm_", i, ".txt"), row.names = FALSE, col.names = FALSE)
}

foreach(i = 1:nsim, .packages = c("splatter", "scran", "scater", "rhdf5")) %dopar% {
  process_simulation(i)
}

stopCluster(cl)
