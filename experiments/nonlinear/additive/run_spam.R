library(SAM)

cross_validate_sam <- function(X, Y, method = samQL, nfolds = 5, nlambda = 30, seed = 110) {
  set.seed(seed)
  n <- nrow(X)
  folds <- sample(rep(1:nfolds, length.out = n))
  lambda_grid <- NULL  # will be assigned after first model fit
  all_errors <- matrix(0, nrow = nlambda, ncol = nfolds)

  for (fold in 1:nfolds) {
    train_idx <- which(folds != fold)
    test_idx  <- which(folds == fold)

    X_train <- X[train_idx, ]
    Y_train <- Y[train_idx]
    X_test  <- X[test_idx, ]
    Y_test  <- Y[test_idx]

    # Fit model on training fold
    fit <- method(X_train, Y_train, nlambda = nlambda)
    if (is.null(lambda_grid)) lambda_grid <- fit$lambda

    # Predict: returns n_test × length(lambda) matrix
    pred_mat <- predict(fit, X_test)$values

    # Compute column-wise MSE
    errors <- colMeans((pred_mat - matrix(Y_test, nrow = length(Y_test), ncol = nlambda))^2)
    all_errors[, fold] <- errors
  }

  # Average CV error across folds
  mean_cv_error <- rowMeans(all_errors)
  best_k <- which.min(mean_cv_error)
  best_lambda <- lambda_grid[best_k]

  list(
    lambda_seq = lambda_grid,
    mean_cv_error = mean_cv_error,
    best_lambda = best_lambda,
    best_index = best_k
  )
}

oracle_choose_k <- function(lambda_vec, func_norm, target = 10) {
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


oracle <- as.integer(Sys.getenv("ADAMP_ORACLE", "1"))
permute <- as.integer(Sys.getenv("ADAMP_PERMUTE", "1"))
snr_list <- c(0.5, 1, 2, 5)
num_rep <- 10
p = 500

num_signals <- 10
corr_list = c(0,0.5,0.9)
out_dir <- "temp_results/nonlinear/additive/"



# ----- Main Loop -----
for (j in seq_along(corr_list)) {
  corr = corr_list[j]
  for (i in seq_along(snr_list)) {
    snr <- snr_list[i]

    for (rep in 0:(num_rep - 1)) {
      # Read data
      data_path <- sprintf("data/simulation/nonlinear_additive_corr%g_snr%g_permute%d_rep%d.csv", corr, snr, permute, rep)
      df <- read.csv(data_path)


      predictor_data <- as.matrix(df[ , !(names(df) %in% c("Y"))])  # all columns except Y
      response_vector <- df$Y

      fit_best <- samQL(predictor_data, response_vector)
      lambda_vec <- fit_best$lambda

      if (oracle == 0) {
          cv_result <- cross_validate_sam(predictor_data, response_vector, method = samQL, nfolds = 5)
          k <- which.min(abs(lambda_vec - cv_result$best_lambda))
      }
      else {
          k <- oracle_choose_k(lambda_vec, fit_best$func_norm, target = num_signals)
      }
      
      norms <- fit_best$func_norm[, k]

      selected_idx <- which(norms != 0)

      if (oracle == 1) {
          if (length(selected_idx) != num_signals) {
              message(sprintf("[WARN] snr=%g rep=%d: selected %d (target 10)", snr, rep, length(selected_idx)))
          }
      }

      out_path <- file.path(out_dir, sprintf("spam_selected_corr%g_snr%g_permute%d_oracle%d_rep%d.csv",corr, snr, permute, oracle, rep))
      
      write.csv(data.frame(var = selected_idx), out_path, row.names = FALSE)

      print(rep)

    }

  }
}
