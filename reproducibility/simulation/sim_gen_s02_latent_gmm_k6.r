# Scenario s02 -- latent Gaussian mixture, K = 6, 20 replicates.
#
# As s01 with K = 6 over the same span of 20, so spacing 4.0 rather than 2.86.
# Also: latent dim 3 -> 5, n_per 750 -> 600 (n = 3,600), within-cluster sd
# 0.50 -> 0.65, B sd 0.20 -> 0.15, gene noise 0.22 -> 0.28, Poisson lift
# 1.8 + 0.70*z -> 1.7 + 0.62*z. All genes retained.

library(doParallel)

numCores <- detectCores() - 4
cl <- makeCluster(max(1, numCores))
registerDoParallel(cl)
nsim <- 20

sim_root <- "/Volumes/SSD/MCW/Research/Codes/Simulation_single_cell/dmvae_sim"
path <- paste0(file.path(sim_root, "s02_latent_gmm_k6"), "/")
.sim_out_root <- Sys.getenv("SIM_OUT_ROOT", unset = "")
if (nzchar(.sim_out_root)) path <- paste0(file.path(.sim_out_root, basename(sub("/+$", "", path))), "/")
if (!dir.exists(path)) {
  dir.create(path, recursive = TRUE)
}

process_simulation <- function(i) {
  set.seed(i + 17001)
  n_per <- 600L
  K <- 6L
  G <- 2000L
  d <- 5L                                   # latent dim (s02 used 3)
  n <- n_per * K

  # K equally spaced centres on a 1D line spanning [-10, 10] (span 20), as in s02
  means <- matrix(0, nrow = K, ncol = d)
  means[, 1] <- seq(-10, 10, length.out = K)

  group <- rep(seq_len(K), each = n_per)
  Z <- matrix(rnorm(n * d, sd = 0.65), nrow = n, ncol = d)   # wider within-cluster sd
  for (k in seq_len(K)) {
    idx <- group == k
    Z[idx, ] <- sweep(Z[idx, , drop = FALSE], 2, means[k, ], "+")
  }

  B <- matrix(rnorm(d * G, sd = 0.15), nrow = d, ncol = G)
  X <- Z %*% B + matrix(rnorm(n * G, sd = 0.28), nrow = n, ncol = G)

  rate <- exp(pmin(4.5, 1.7 + 0.62 * scale(X)))
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
