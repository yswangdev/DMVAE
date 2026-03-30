library(cluster)
library(foreach)
library(doParallel)

# Set up parallel backend
n_cores <- parallel::detectCores()
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Parameters
args <- commandArgs(trailingOnly = TRUE)

input_datafile <- args[1]
output_datafile <- args[2]
n_sim <- 20
B <- 20  # Number of bootstrap replicates (keep small)
max_clusters <- 10

# Function to compute Gap Statistic for one file
run_gap_stat <- function(filename, B, max_clusters) {
  data <- read.table(filename, header = TRUE)
  
  set.seed(123)
  clusGap(data, FUN = kmeans, nstart = 10, K.max = max_clusters, B = B)
}

# Parallel processing
gap_results <- foreach(i = 1:n_sim, .packages = c("cluster")) %dopar% {
  filename <- paste0(input_datafile, "simnorm_", i, ".txt")
  gap_stat <- run_gap_stat(filename, B, max_clusters)
  list(index = i, optimal_k = which.max(gap_stat$Tab[, "gap"]), gap_stat = gap_stat)
}

# Stop cluster
stopCluster(cl)

# Summarize optimal k for each simulation
optimal_k_list <- sapply(gap_results, function(x) x$optimal_k)
print(optimal_k_list)

save(gap_results, file = paste0(output_datafile, "gap_stat_results.RData"))