# This file is just a place holder (not used anymore), simulation datasets are all generated from python code

.component_functions_nonadd <- function(X) {
  list(
    X[, 1] * X[, 2],                  # f1: interaction
    (X[, 3])^2,                       # f2
    exp(X[, 4]),                      # f3
    X[, 5],                           # f4
    log(abs(X[, 6]) + 1),             # f5
    pmax(0, X[, 7]),                  # f6 (ReLU-like)
    as.numeric(X[, 8] > 0),           # f7 (step)
    X[, 9],                           # f8
    X[, 10]                           # f9
  )
}

# Toeplitz covariance corr^{|i-j|}
.toeplitz_cov <- function(M, corr) {
  idx <- 0:(M - 1)
  outer(idx, idx, function(i, j) corr^abs(i - j))
}

SimuFriedmanNonAdditive <- function(
  N, M, N1,
  snr = 1, seed = 12, corr = 0.9,
  interaction_snr = 5,
  permute_cov = FALSE
) {
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Please install 'MASS': install.packages('MASS')")
  }

  set.seed(seed)

  # Build Toeplitz covariance and (optionally) permute it
  Sigma <- .toeplitz_cov(M, corr)
  perm <- NULL
  if (permute_cov) {
    perm <- sample.int(M)
    Sigma <- Sigma[perm, perm]
  }

  # ---- Training data ----
  X <- MASS::mvrnorm(n = N, mu = rep(0, M), Sigma = Sigma)
  E <- rnorm(N)

  comps  <- .component_functions_nonadd(X)               # 9 components
  vars   <- vapply(comps, var, numeric(1))
  # betas ~ Uniform(2,3), with interaction term scaled by interaction_snr
  betas  <- runif(9, min = 2, max = 3) * c(interaction_snr, rep(1, 8))
  scaled_betas <- betas / sqrt(vars)
  f_true <- Reduce(`+`, Map(`*`, scaled_betas, comps))

  noise_var <- var(f_true) / snr
  Y <- f_true + rnorm(N, sd = sqrt(noise_var))

  # ---- Test data ----
  X1 <- MASS::mvrnorm(n = N1, mu = rep(0, M), Sigma = Sigma)
  E1 <- rnorm(N1)

  comps1 <- .component_functions_nonadd(X1)
  vars1  <- vapply(comps1, var, numeric(1))
  # test betas ~ Uniform(0,1) (no interaction multiplier)
  betas1 <- runif(9, min = 0, max = 1)
  scaled_betas1 <- betas1 / sqrt(vars1)
  f_true1 <- Reduce(`+`, Map(`*`, scaled_betas1, comps1))

  noise_var1 <- var(f_true1) / snr
  Y1 <- f_true1 + rnorm(N1, sd = sqrt(noise_var1))

  # Return (include permutation used if requested)
  list(X = X, Y = Y, X1 = X1, Y1 = Y1, perm = perm)
}

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

SimuFriedmanAdditive <- function(N, M, N1, snr = 1, seed = 12, corr = 0.9) {
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