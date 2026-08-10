# Scenario s01 -- latent Gaussian mixture, K = 8, 20 replicates.
#
# K equally spaced collinear Gaussian means in a low-dimensional latent space,
# mapped linearly to G = 2,000 genes and converted to counts by a Poisson lift, so
# log1p(count) stays approximately Gaussian. All genes retained.

library(doParallel)

numCores <- detectCores() - 4
cl <- makeCluster(max(1, numCores))
registerDoParallel(cl)
nsim <- 20

sim_root <- "/Volumes/SSD/MCW/Research/Codes/Simulation_single_cell/dmvae_sim"
path <- paste0(file.path(sim_root, "s01_latent_gmm_k8"), "/")
.sim_out_root <- Sys.getenv("SIM_OUT_ROOT", unset = "")
if (nzchar(.sim_out_root)) path <- paste0(file.path(.sim_out_root, basename(sub("/+$", "", path))), "/")
if (!dir.exists(path)) {
  dir.create(path, recursive = TRUE)
}

process_simulation <- function(i) {
  set.seed(i + 12001)
  n_per <- 750L
  K <- 8L
  G <- 2000L
  n <- n_per * K

  means <- matrix(0, nrow = K, ncol = 3)
  means[, 1] <- seq(-10, 10, length.out = K)

  group <- rep(seq_len(K), each = n_per)
  Z <- matrix(rnorm(n * 3, sd = 0.5), nrow = n, ncol = 3)
  for (k in seq_len(K)) {
    idx <- group == k
    Z[idx, ] <- sweep(Z[idx, , drop = FALSE], 2, means[k, ], "+")
  }

  B <- matrix(rnorm(3 * G, sd = 0.20), nrow = 3, ncol = G)
  X <- Z %*% B + matrix(rnorm(n * G, sd = 0.22), nrow = n, ncol = G)

  rate <- exp(pmin(4.5, 1.8 + 0.70 * scale(X)))
  counts <- matrix(rpois(length(rate), as.vector(rate)), nrow = n, ncol = G)
  counts_data <- counts

  logexpr <- log1p(counts)
  data <- as.data.frame(logexpr)

  if (sum(apply(data, 2, function(x) sum(x) == 0)) > 0) {
    data <- data[, -which(apply(data, 2, function(x) sum(x) == 0))]
  }
  scale_data <- as.data.frame(scale(data))

  sim_meta <- data.frame(cluster = as.integer(group))
  write.table(sim_meta, paste0(path, "simmeta_", i, ".txt"), row.names = FALSE, col.names = FALSE)
  write.table(counts_data, paste0(path, "simcounts_", i, ".csv"), sep = ",", row.names = FALSE, col.names = FALSE)

  # Raw-count h5 (X = cells x genes, Y = labels; ALL genes) for scAce/ADClust/scVI (github interface)
  h5f <- paste0(path, "sim_", i, ".h5")
  if (file.exists(h5f)) file.remove(h5f)
  rhdf5::h5createFile(h5f)
  .Xc <- t(counts_data); storage.mode(.Xc) <- "integer"
  rhdf5::h5write(.Xc, h5f, "X")
  rhdf5::h5write(as.integer(sim_meta[[1]]), h5f, "Y")
  rhdf5::H5close()

  write.table(data, paste0(path, "simdata_", i, ".txt"), row.names = FALSE, col.names = FALSE)

  data_norm <- as.data.frame(lapply(scale_data, function(x) {
    if (is.numeric(x)) (x - min(x)) / (max(x) - min(x)) else x
  }))
  write.table(data_norm, paste0(path, "simnorm_", i, ".txt"), row.names = FALSE, col.names = FALSE)
}

foreach(i = 1:nsim, .packages = "rhdf5",
        .export = c("process_simulation", "path")) %dopar% {
  process_simulation(i)
}

stopCluster(cl)
