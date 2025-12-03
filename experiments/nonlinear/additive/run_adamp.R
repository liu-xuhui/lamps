library(earth)
library(stringr)
library(future)
library(future.apply)
library(RhpcBLASctl)

blas_set_num_threads(1)


f1_score <- function(selected, true_support, p) {
  selected <- as.integer(selected)
  true_support <- as.integer(true_support)

  tp <- length(intersect(selected, true_support))
  fp <- length(setdiff(selected, true_support))
  fn <- length(setdiff(true_support, selected))

  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  if (precision + recall == 0) return(0)

  return(2 * precision * recall / (precision + recall))
}

linear_reg <- function(X, Y, X1) {
  # Fit linear regression without intercept
  model <- lm(Y ~ . - 1, data = as.data.frame(X))  # "-1" removes the intercept
  
  # Predict on new data
  preds <- as.numeric(predict(model, newdata = as.data.frame(X1)))
  
  return(preds)
}

mars_reg <- function(X, Y, X1) {

  model <- earth::earth(
    x = X,
    y = Y,
    trace = 0     # suppress output
  )

  preds <- as.numeric(predict(model, newdata = X1))
  return(preds)
}

indept_sample_array <- function(probability) {
  n <- length(probability)
  repeat {
    sel <- runif(n) < probability
    if (any(sel)) return(which(sel))
  }
}

# Build one minipatch (obs + features)
buildMP_indept <- function(X, Y, n_ratio, m_ratio, prob_I = NULL, prob_F = NULL, delta = 1) {
  N <- nrow(X)
  M <- ncol(X)
  n <- round(n_ratio * N)
  m <- round(m_ratio * M)

  # observations
  if (is.null(prob_I)) {
    idx_I <- sort(sample.int(N, size = n, replace = FALSE))
  } else {
    idx_I <- sort(sample.int(N, size = n, replace = FALSE, prob = prob_I))
  }

  # features
  if (is.null(prob_F)) {
    # Bernoulli thinning with p = m_ratio
    idx_F <- indept_sample_array(rep(m_ratio, M))
  } else {
    idx_F <- indept_sample_array(prob_F)
  }

  x_mp <- X[idx_I, idx_F, drop = FALSE]
  y_mp <- Y[idx_I]
  list(idx_I = idx_I, idx_F = idx_F, x_mp = x_mp, y_mp = y_mp)
}

# Predict on stacked data using B minipatches
predictMP_indept <- function(X, Y, X1, n_ratio, m_ratio, B, fit_func,
                             prob_I = NULL, prob_F = NULL, delta = 1) {
  N  <- nrow(X)
  M  <- ncol(X)
  N1 <- nrow(X1)

  in_mp_obs     <- matrix(FALSE, nrow = B, ncol = N)
  in_mp_feature <- matrix(FALSE, nrow = B, ncol = M)
  preds_mat     <- matrix(NA_real_, nrow = B, ncol = N + N1)

  Xstack <- rbind(X, X1)

  for (b in seq_len(B)) {
    mp <- buildMP_indept(X, Y, n_ratio, m_ratio, prob_I, prob_F, delta)
    preds_mat[b, ] <- fit_func(mp$x_mp, mp$y_mp, Xstack[, mp$idx_F, drop = FALSE])

    in_mp_obs[b, mp$idx_I] <- TRUE
    in_mp_feature[b, mp$idx_F] <- TRUE
  }

  list(predictions = preds_mat, in_mp_obs = in_mp_obs, in_mp_feature = in_mp_feature)
}

# Helper: LOCO prediction mean for sample i, feature j
get_loco <- function(i, j, in_mp_feature, in_mp_obs, predictions_train) {
  # MPs that exclude both sample i and feature j
  keep <- which(!in_mp_feature[, j] & !in_mp_obs[, i])
  if (length(keep) == 0) return(NA_real_)
  mean(predictions_train[keep, i])
}

predictMP_indept_parallel <- function(X, Y, X1, n_ratio, m_ratio, B, fit_func,
                                      prob_I = NULL, prob_F = NULL, delta = 1,
                                      workers = max(1, parallel::detectCores() - 1)) {
  N  <- nrow(X)
  M  <- ncol(X)
  N1 <- nrow(X1)

  # Stack once (same as your original)
  Xstack <- rbind(X, X1)

  # Use multisession (works on Windows). On Linux/macOS, you can try multicore.
  oplan <- plan()
  on.exit(plan(oplan), add = TRUE)
  plan(multisession, workers = workers)

  # IMPORTANT: ensure reproducible RNG across parallel workers
  # (keeps distribution identical, though draw order differs)
  res <- future_lapply(
    X = seq_len(B),
    FUN = function(b) {
      mp <- buildMP_indept(X, Y, n_ratio, m_ratio, prob_I, prob_F, delta)
      # Build row indicators INSIDE the worker to avoid a second combining loop
      obs_row   <- logical(N);  obs_row[mp$idx_I] <- TRUE
      feat_row  <- logical(M);  feat_row[mp$idx_F] <- TRUE
      pred_row  <- fit_func(mp$x_mp, mp$y_mp, Xstack[, mp$idx_F, drop = FALSE])
      list(pred = pred_row, obs = obs_row, feat = feat_row)
    },
    future.seed = TRUE  # reproducible parallel RNG
  )

  # Combine results
  preds_mat     <- do.call(rbind, lapply(res, `[[`, "pred"))
  in_mp_obs     <- do.call(rbind, lapply(res, `[[`, "obs"))
  in_mp_feature <- do.call(rbind, lapply(res, `[[`, "feat"))

  list(predictions = preds_mat,
       in_mp_obs = in_mp_obs,
       in_mp_feature = in_mp_feature)
}

# Score minipatch ensemble + LOOCV & LOCO deltas
MPRegFeatureScore_indept <- function(X, Y, X1, Y1, n_ratio, m_ratio, K, fit_func,
                                     prob_I = NULL, prob_F = NULL, delta = 1) {
  N <- nrow(X)
  M <- ncol(X)

#   pred_obj <- predictMP_indept(X, Y, rbind(X, X1), n_ratio, m_ratio, K, fit_func,
#                                prob_I = prob_I, prob_F = prob_F, delta = delta)
  
  pred_obj <- predictMP_indept_parallel(X, Y, rbind(X, X1), n_ratio, m_ratio, K, fit_func,
                               prob_I = prob_I, prob_F = prob_F, delta = delta, workers = 8)

  predictions     <- pred_obj$predictions
  in_mp_obs       <- pred_obj$in_mp_obs
  in_mp_feature   <- pred_obj$in_mp_feature

  predictions_train <- predictions[, seq_len(N), drop = FALSE]
  predictions_test  <- predictions[, (N + 1):(N + nrow(X1)), drop = FALSE]

  # ensemble means
  mean_train <- colMeans(predictions_train)
  mean_test  <- colMeans(predictions_test)

  mse_train <- mean((Y  - mean_train)^2)
  mse_test  <- mean((Y1 - mean_test)^2)

  # LOO per sample i: exclude any MP that contains sample i
  b_keep_list <- lapply(seq_len(N), function(i) which(!in_mp_obs[, i]))
  resids_LOO <- vapply(seq_len(N), function(i) {
    keep <- b_keep_list[[i]]
    if (length(keep) == 0) return(NA_real_)
    (abs(Y[i] - mean(predictions_train[keep, i])))^2
  }, numeric(1))

  LOO_sd <- mean(
  vapply(seq_len(ncol(predictions_train)), function(i) {
    keep <- which(!in_mp_obs[, i])
    if (length(keep) == 0) return(NA_real_)
    sd(predictions_train[keep, i], na.rm = TRUE)
  }, numeric(1)),
  na.rm = TRUE
  )
  LOO_mean <- mean(vapply(seq_len(N), function(i) {
    keep <- b_keep_list[[i]]
    if (length(keep) == 0) return(NA_real_) else mean(predictions_train[keep, i])
  }, numeric(1)), na.rm = TRUE)

  # LOCO-LOO residuals by (i,j)
  # Build data.frame of (i,j, resid_loco, resid_loo, zz)
  # We can compute Delta[j] = mean over i of (resid_loco - resid_loo)
  Delta <- numeric(M)

  for (j in seq_len(M)) {
    zz_j <- vapply(seq_len(N), function(i) {
      loco_mean <- get_loco(i, j, in_mp_feature, in_mp_obs, predictions_train)
      if (is.na(loco_mean) || is.na(resids_LOO[i])) return(NA_real_)
      resid_loco <- (abs(Y[i] - loco_mean))^2
      resid_loco - resids_LOO[i]
    }, numeric(1))
    Delta[j] <- mean(zz_j, na.rm = TRUE)
  }

  list(
    Delta     = Delta,
    mse_train = mse_train,
    mse_test  = mse_test,
    loo       = mean(resids_LOO, na.rm = TRUE),
    loo_std   = LOO_sd,
    loo_mean  = LOO_mean,
    m         = sum(in_mp_feature[1, ]),        # number of features in first MP
    last_pred = predictions[, ncol(predictions)]
  )
}

# Epoch-tuned weighting and feature-sampling probabilities
indept_weight_sample_epochtuned <- function(X, Y, X1, Y1, n_ratio, m_ratio, K,
                                            fit_func, delta, max_iter) {
  M <- ncol(X)
  kk <- 0L
  prob_I <- NULL
  prob_F <- NULL
  res <- list()

  K_list <- if (length(K) == 1L) rep(K, max_iter) else K

  weight <- rep(1 / M, M)  # initialize for possible return
  SelectedFeatures <- integer(0)

  while (kk < max_iter) {
    cur <- MPRegFeatureScore_indept(X, Y, X1, Y1, n_ratio, m_ratio, K_list[kk + 1L],
                                    fit_func, prob_I, prob_F, delta)
    res[[as.character(kk)]] <- cur

    if (kk > 0) {
      prev <- res[[as.character(kk - 1L)]]
      if (cur$loo >= prev$loo) {
        # revert the last (worse) step
        res[[as.character(kk)]] <- NULL
        break
      }
    }

    # Update sampling probabilities for features
    Delta <- cur$Delta
    Delta_shift <- Delta - min(Delta)
    weight_sort <- sort(Delta_shift, decreasing = TRUE)

    total_sum <- sum(weight_sort)
    running_sum <- 0
    target_index <- 0L
    m <- M * m_ratio

    for (i in seq_len(M)) {
      running_sum <- running_sum + weight_sort[i]
      # careful i+1 indexing; if i == M, break
      if (i < M) {
        if (total_sum - running_sum > (m / delta - i - 1) * weight_sort[i + 1]) {
          target_index <- i
          break
        }
      } else {
        target_index <- i
      }
    }

    if (target_index >= M) {
      delta_bar <- 0
    } else {
      denom <- (m / delta - (target_index + 1))
      delta_bar <- if (denom > 0) sum(weight_sort[(target_index + 1):M]) / denom else 0
    }

    weight <- pmin(Delta_shift, delta_bar)
    sum_weight <- sum(weight)
    if (sum_weight <= 0) {
      prob_F <- rep(m / M, M)  # fallback to uniform mass m
    } else {
      prob_F <- as.numeric(m * weight / sum_weight)
    }

    # Store extras
    res[[as.character(kk)]]$prob_F <- prob_F
    res[[as.character(kk)]]$weight <- weight

    kk <- kk + 1L
  }

  # Select features (same rule: weight > 1/M)
  SelectedFeatures <- which(weight > 1 / M)

  # Return both the epoch records and selection for convenience
  attr(res, "SelectedFeatures") <- SelectedFeatures
  res
}

## -----------------------------
## Synthetic data generator
## -----------------------------

# Component functions on matrix X (expects at least 10 columns)
.component_functions <- function(X) {
  list(
    sin(pi * X[, 1]),
    (X[, 2])^2,
    X[, 3],
    X[, 4],
    exp(X[, 5]),
    log(abs(X[, 6]) + 1),
    pmax(0, X[, 7]),
    as.numeric(X[, 8] > 0),
    X[, 9],
    X[, 10]
  )
}

# Toeplitz covariance corr^{|i-j|}
.toeplitz_cov <- function(M, corr) {
  idx <- 0:(M - 1)
  outer(idx, idx, function(i, j) corr^abs(i - j))
}

SimuFriedmanAdditive <- function(N, M, N1, snr = 1, seed = 42, corr = 0.9) {
  set.seed(seed)
  cov <- .toeplitz_cov(M, corr)

  # X ~ N(0, cov)
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Please install 'MASS' for mvrnorm: install.packages('MASS')")
  }
  X  <- MASS::mvrnorm(N, mu = rep(0, M), Sigma = cov)
  E  <- rnorm(N)

  comps <- .component_functions(X)
  variances <- vapply(comps, var, numeric(1))
  betas <- runif(10, 2, 3)
  scaled_betas <- betas / sqrt(variances)
  f_true <- Reduce(`+`, Map(function(b, c) b * c, scaled_betas, comps))

  noise_var <- var(f_true) / snr
  Y <- f_true + sqrt(noise_var) * E

  # Test set
  X1 <- MASS::mvrnorm(N1, mu = rep(0, M), Sigma = cov)
  E1 <- rnorm(N1)

  comps1 <- .component_functions(X1)
  variances1 <- vapply(comps1, var, numeric(1))
  betas1 <- runif(10, 2, 3)
  scaled_betas1 <- betas1 / sqrt(variances1)
  f_true1 <- Reduce(`+`, Map(function(b, c) b * c, scaled_betas1, comps1))

  noise_var1 <- var(f_true1) / snr
  Y1 <- f_true1 + sqrt(noise_var1) * E1

  list(X = X, Y = Y, X1 = X1, Y1 = Y1)
}




N  <- 200
M  <- 500
N1 <- 50

max_iter <- 5

perm <- 1

func_name <- "mr"
fit_funcs <- list(
  linear = linear_reg, mr = mars_reg
)
fit_func <- fit_funcs[[func_name]]

n_ratio <- 0.4
m_ratio <- 0.12
delta   <- 0.8
K       <- c(5787, 5787, 5787, 5787, 5787)
number_signals <- 10

snr_list <- c(0.5, 1, 2, 5)
num_rep <- 10
p <- 500
corr_list <- c(0, 0.5, 0.9)
oracle_list <- c(0, 1)

# result <- data.frame(snr = snr_list, f1 = rep(NA, length(snr_list)))
out_dir <- "temp_results/nonlinear/additive"

sim <- SimuFriedmanAdditive(N, M, N1, snr = 5, seed = 42, corr = corr)
# useless
X1 <- sim$X1
Y1 <- sim$Y1


pb <- txtProgressBar(min = 0, max = length(corr_list) * length(snr_list) * num_rep, style = 3)
counter <- 0
# ----- Main Loop -----

for (j in seq_along(corr_list)) {
    corr <- corr_list[j]
    for (i in seq_along(snr_list)) {
        snr <- snr_list[i]
        f1_vec <- c()

        for (rep in 0:(num_rep - 1)) {
            data_path <- sprintf("data/simulation/nonlinear_additive_corr%g_snr%g_permute%d_rep%d.csv", corr, snr, permute, rep)
            df <- read.csv(data_path)

            predictor_data <- as.matrix(df[ , !(names(df) %in% c("Y"))])  # all columns except Y
            response_vector <- df$Y

            res <- indept_weight_sample_epochtuned(
                predictor_data, response_vector, X1, Y1,
                n_ratio = n_ratio,
                m_ratio = m_ratio,
                K       = K,
                fit_func = fit_func,
                delta    = delta,
                max_iter = max_iter
            )

            for (q in seq_along(oracle_list)) {

                oracle <- oracle_list[q]
                stopifnot(length(res) >= 1L)

                key_names <- sort(names(res))      
                num_epochs <- length(key_names)

                choose_key_name <- NULL

                if (num_epochs == 1L) {
                choose_key_name <- key_names[1L]
                } else if (num_epochs == max_iter) {
                choose_key_name <- key_names[num_epochs]
                } else {
                choose_key_name <- key_names[num_epochs - 1L]
                }

                prob_F_use <- res[[choose_key_name]]$prob_F
                if (is.null(prob_F_use)) stop("Selected epoch has no prob_F stored.")

                if (oracle == 0) {
                    nonlinear_select1 <- which(prob_F_use >= delta * 0.5)
                }
                else {
                    nonlinear_select1 <- order(prob_F_use, decreasing = TRUE)[1:number_signals]
                }

                
                out_path <- file.path(out_dir, sprintf("selected_vars_add_corr%g_oracle%d_snr%g_permute%d_rep%d.csv",corr, oracle, snr, permute, rep))
                write.csv(data.frame(var = nonlinear_select1), out_path, row.names = FALSE)

            }

            counter <- counter + 1
            setTxtProgressBar(pb, counter)
        }
    }
}


