load("/scratch/g/chlin/Yushu/results/Comparison_methods/gap_stat/4_clus/gap_stat_results.RData")
mean(sapply(1:20, function(i) gap_results[[i]]$optimal_k))
elbow <- readRDS("/scratch/g/chlin/Yushu/results/Comparison_methods/select_k/elbow/all_results_elbow.rds")

library(data.table)
acc <- fread("/scratch/g/chlin/Yushu/results/Comparison_methods/PBMC/gmm/r6/k5/gmm_acc.txt")
mean(acc$V1)
sd(acc$V1)

folders <- sprintf("i%02d", 1:20)

# loop through, read file, compute mean, collect
means <- sapply(folders, function(f) {
  acc <- fread(paste0("/scratch/g/chlin/Yushu/results/Comparison_methods/gmm/4_clus/k3/", f, "/gmm_acc.txt"))
  mean(acc$V1, na.rm = TRUE)
})

# overall mean of means
grand_mean <- mean(means)
grand_mean
