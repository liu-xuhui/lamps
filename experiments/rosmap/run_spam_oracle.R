library(SAM)

oracle_choose_k <- function(lambda_vec, func_norm, target = 30) {
  # func_norm: matrix of size p x K (features x lambdas)
  nz_counts <- colSums(func_norm != 0)

  # 1) Exact match
  exact <- which(nz_counts == target)
  if (length(exact) > 0) {
    # prefer stronger regularization among exact matches
    k <- exact[ which.max(lambda_vec[exact]) ]
    return(k)
  }

  # 2) Closest match (minimize |count - target|)
  d <- abs(nz_counts - target)
  ties <- which(d == min(d))
  # prefer larger lambda among ties
  k <- ties[ which.max(lambda_vec[ties]) ]
  return(k)
}


oracle <- 1
features_to_choose <- 30
n_splits <- 10


## Paths
data_dir <- file.path("data", "rosmap")
out_dir  <- file.path("results", "rosmap")

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

## Storage: 50 x 10 matrix
## Each column i holds top-50 selected feature indices for split i (0-indexed)
selected_mat_0idx <- matrix(NA_integer_, nrow = features_to_choose + 10, ncol = n_splits)
colnames(selected_mat_0idx) <- sprintf("split_%02d", seq_len(n_splits))
rownames(selected_mat_0idx) <- paste0("rank_", seq_len(features_to_choose + 10))

## Loop over splits
for (i in seq_len(n_splits)) {

  train_file <- file.path(data_dir, sprintf("rosmap_train_split_%02d.csv", i))
  if (!file.exists(train_file)) {
    stop("Training file not found: ", train_file)
  }

  train_df <- read.csv(train_file, check.names = FALSE)

  if (!("Y" %in% names(train_df))) stop("Column 'Y' not found in: ", train_file)

  Y_train <- train_df$Y
  X_train <- as.matrix(train_df[, setdiff(names(train_df), "Y"), drop = FALSE])


  ## Sanity check dimensions
  stopifnot(nrow(X_train) == length(Y_train))

  ## Seed so results are reproducible per split & per base model
  set.seed(110 + i)

  fit_best <- samQL(X_train, Y_train)
  lambda_vec <- fit_best$lambda
  k <- oracle_choose_k(lambda_vec, fit_best$func_norm, target = features_to_choose)

  norms <- fit_best$func_norm[, k]

  selected_idx <- order(norms, decreasing = TRUE)[seq_len(k)] - 1L

  selected_mat_0idx[1:length(selected_idx), i] <- selected_idx

  print(i)

  
}

out_file <- file.path(out_dir,"rosmap_spam.csv")

write.csv(selected_mat_0idx, out_file, row.names = TRUE)

cat("Saved selected feature indices to:\n", out_file, "\n", sep = "")




