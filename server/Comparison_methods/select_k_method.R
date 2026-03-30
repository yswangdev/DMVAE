library(cluster)
library(factoextra)
library(NbClust)
library(fpc)
library(foreach)
library(doParallel)

# Setup
args <- commandArgs(trailingOnly = TRUE)
input_datafile <- args[1]      # prefix
output_datafile <- args[2]     
method <- args[3]              # one of: gap, elbow, silhouette, prediction, nbclust
n_sim <- 20
max_clusters <- 10

# Set up parallel backend
n_cores <- parallel::detectCores()
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Method runners
run_gap_stat <- function(data, i) {
  set.seed(123)
  gap_stat <- clusGap(data, FUN = kmeans, nstart = 10, K.max = max_clusters, B = 20)
  saveRDS(gap_stat, file = paste0(output_datafile, "gap_stat_", i, ".rds"))
  pdf(paste0(output_datafile, "gap_plot_", i, ".pdf"))
  fviz_gap_stat(gap_stat) + ggtitle(paste("Gap Statistic - Sim", i))
  dev.off()
  which.max(gap_stat$Tab[, "gap"])
}

run_elbow <- function(data, i) {
  pdf(paste0(output_datafile, "elbow_", i, ".pdf"))
  print(fviz_nbclust(data, kmeans, method = "wss") +
          ggtitle(paste("Elbow Method - Sim", i)))
  dev.off()
}

run_silhouette <- function(data, i) {
  pdf(paste0(output_datafile, "silhouette_", i, ".pdf"))
  print(fviz_nbclust(data, kmeans, method = "silhouette") +
          ggtitle(paste("Silhouette Method - Sim", i)))
  dev.off()
}

run_prediction_strength <- function(data, i) {
  set.seed(123)
  ps <- prediction.strength(data, Gmin = 2, Gmax = max_clusters, M = 5, nstart = 20)
  saveRDS(ps, file = paste0(output_datafile, "prediction_strength_", i, ".rds"))
  pdf(paste0(output_datafile, "prediction_strength_", i, ".pdf"))
  plot(ps$mean.pred, type = "l", main = paste("Prediction Strength - Sim", i),
       xlab = "Number of clusters", ylab = "Mean Prediction Strength")
  dev.off()
}

run_nbclust <- function(data, i) {
  set.seed(123)
  nc <- NbClust(data, method = 'complete', index = 'all')
  saveRDS(nc, file = paste0(output_datafile, "nbclust_", i, ".rds"))
  best_k <- nc$Best.nc[1, ]
  write.table(best_k, file = paste0(output_datafile, "nbclust_k_", i, ".txt"), row.names = FALSE, col.names = FALSE)
  return(best_k)
}

# Main loop
results <- foreach(i = 1:n_sim, .packages = c("cluster", "factoextra", "fpc", "NbClust")) %dopar% {
  filename <- paste0(input_datafile, "simnorm_", i, ".txt")
  data <- read.table(filename, header = TRUE)

  if (method == "gap") {
    run_gap_stat(data, i)
  } else if (method == "elbow") {
    run_elbow(data, i)
  } else if (method == "silhouette") {
    run_silhouette(data, i)
  } else if (method == "prediction") {
    run_prediction_strength(data, i)
  } else if (method == "nbclust") {
    run_nbclust(data, i)
  } else {
    stop("Invalid method selected.")
  }
}

stopCluster(cl)
saveRDS(results, file = paste0(output_datafile, "all_results_", method, ".rds"))