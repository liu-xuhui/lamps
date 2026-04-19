library(RhpcBLASctl)
blas_set_num_threads(1)

## -------------------------------------------------------------------
## Load AdaMP code
## -------------------------------------------------------------------
library(here)
root <- here::here()

source(file.path(root, "src", "R", "adamp", "adamp_main.R"))
source(file.path(root, "src", "R", "adamp", "fit_functions.R"))
source(file.path(root, "src", "R", "adamp", "simulation_functions.R"))  # optional / placeholder

## -------------------------------------------------------------------
## Fixed experiment settings
## -------------------------------------------------------------------
N  <- 200
M  <- 500
N1 <- 50

max_iter <- 5
delta    <- 0.8

permute <- as.integer(Sys.getenv("ADAMP_PERMUTE", "0"))
corr    <- 0.5

# base model: MARS
func_name <- "mr"
fit_funcs <- list(
  linear = linear_reg,
  mr     = mars_reg,
  sp     = spam_reg
)
fit_func <- fit_funcs[[func_name]]
stopifnot(!is.null(fit_func))

# only non-oracle selection here (as requested)
oracle <- 0L

# true signal set for F1
signal_set <- 1:10

# experiment loops
snr_list <- c(0.5, 1, 2, 5)
num_rep  <- 10
rep_list <- 0:(num_rep - 1)

# tuning grid (9 combos)
n_ratio_list <- c(0.3, 0.4, 0.5)
m_ratio_list <- c(0.1, 0.15, 0.2)
grid <- expand.grid(
  n_ratio = n_ratio_list,
  m_ratio = m_ratio_list,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

## -------------------------------------------------------------------
## Output dirs (as required)
## -------------------------------------------------------------------
base_out_dir <- file.path(
  root, "temp_results", "hyperparam_tuning",
  sprintf("nonlinear_additive_corr0.5_permute%d", permute)
)
best_out_dir <- file.path(base_out_dir, "best")
grid_out_dir <- file.path(base_out_dir, "grid")

dir.create(base_out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(best_out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(grid_out_dir, recursive = TRUE, showWarnings = FALSE)

## -------------------------------------------------------------------
## Helpers (borrowed/merged from your hyperparam_tune.R)
## -------------------------------------------------------------------
calc_K_one <- function(n_ratio, m_ratio) {
  denom1 <- (1 - n_ratio) * (m_ratio^2)
  denom2 <- (1 - n_ratio) * m_ratio
  if (denom1 <= 0 || denom2 <= 0) stop("Invalid (n_ratio, m_ratio)")
  k1 <- 50  / denom1
  k2 <- 200 / denom2
  as.integer(ceiling(max(k1, k2)))
}

extract_final_loo <- function(res) {
  last_obj <- res[[length(res)]]
  if (!is.null(last_obj$loo)) return(last_obj$loo)
  stop("Could not extract final LOO from res.")
}

extract_selected_features_nonoracle <- function(res, max_iter, delta) {
  stopifnot(length(res) >= 1L)
  key_names  <- sort(names(res))
  num_epochs <- length(key_names)

  # match your run_adamp.R rule: if >1 epoch, use (num_epochs - 1)
  if (num_epochs == 1L) {
    choose_key_name <- key_names[1L]
  } else {
    choose_key_name <- key_names[num_epochs - 1L]
  }

  prob_F_use <- res[[choose_key_name]]$prob_F
  if (is.null(prob_F_use)) stop("Selected epoch has no prob_F stored.")

  which(prob_F_use >= delta * 0.5)
}

f1_from_selected <- function(selected_idx, signal_set) {
  selected_idx <- unique(as.integer(selected_idx))
  signal_set   <- unique(as.integer(signal_set))

  tp <- length(intersect(selected_idx, signal_set))
  fp <- length(setdiff(selected_idx, signal_set))
  fn <- length(setdiff(signal_set, selected_idx))

  denom <- 2 * tp + fp + fn
  if (denom == 0) return(0)
  (2 * tp) / denom
}

fmt_ratio_for_filename <- function(x) {
  # keep filenames clean: 0.3 -> 0.3, 0.15 -> 0.15 (no trailing zeros enforced)
  # If you prefer 2 digits always, replace with formatC(x, format="f", digits=2)
  format(x, trim = TRUE, scientific = FALSE)
}

write_selected_csv <- function(selected_idx, out_path) {
  write.csv(data.frame(var = selected_idx), out_path, row.names = FALSE)
}

## -------------------------------------------------------------------
## Main: for each (snr, rep), run 9 combos; pick best by LOO
## Also: for each snr, write grid summary (avg loo + avg f1 across reps)
## -------------------------------------------------------------------
total_runs <- length(snr_list) * length(rep_list) * nrow(grid)
pb <- txtProgressBar(min = 0, max = total_runs, style = 3)
counter <- 0

for (snr in snr_list) {

  # store per-snr summary across reps for each grid point
  grid_summary <- data.frame(
    n_ratio = grid$n_ratio,
    m_ratio = grid$m_ratio,
    avg_loo_final = NA_real_,
    avg_f1 = NA_real_,
    stringsAsFactors = FALSE
  )

  # matrices to accumulate rep-wise metrics: rows=grid, cols=reps
  loo_mat <- matrix(NA_real_, nrow = nrow(grid), ncol = length(rep_list))
  f1_mat  <- matrix(NA_real_, nrow = nrow(grid), ncol = length(rep_list))

  for (rr in seq_along(rep_list)) {
    rep <- rep_list[rr]

    # load data once per (snr, rep)
    data_path <- file.path(
      root, "data", "simulation",
      sprintf("nonlinear_additive_corr%g_snr%g_permute%d_rep%d.csv",
              corr, snr, permute, rep)
    )
    if (!file.exists(data_path)) stop("Data file not found: ", data_path)

    df <- read.csv(data_path)
    X <- as.matrix(df[, !(names(df) %in% c("Y"))])
    Y <- scale(df$Y)

    # placeholder sim (kept consistent with your scripts)
    sim <- SimuFriedmanAdditive(N, M, N1, snr = snr, seed = 42, corr = corr)
    X1 <- sim$X1
    Y1 <- sim$Y1

    # track best
    best_loo <- Inf
    best_selected <- integer(0)
    best_n <- NA_real_
    best_m <- NA_real_

    for (gg in seq_len(nrow(grid))) {
      n_ratio <- grid$n_ratio[gg]
      m_ratio <- grid$m_ratio[gg]

      # control AdaMP randomness per seed (rep)
      set.seed(rep)

      K_one <- calc_K_one(n_ratio, m_ratio)
      K_vec <- rep(K_one, max_iter)

      res <- indept_weight_sample_epochtuned(
        X, Y, X1, Y1,
        n_ratio  = n_ratio,
        m_ratio  = m_ratio,
        K        = K_vec,
        fit_func = fit_func,
        delta    = delta,
        max_iter = max_iter
      )

      loo_final <- extract_final_loo(res)
      selected_idx <- extract_selected_features_nonoracle(res, max_iter, delta)
      f1 <- f1_from_selected(selected_idx, signal_set)

      # save per-seed, per-grid selected features
      out_path_grid <- file.path(
        base_out_dir,
        sprintf(
          "adammars_selected_corr%g_snr%g_permute%d_oracle%d_rep%d_n%s_m%s.csv",
          corr, snr, permute, oracle, rep,
          fmt_ratio_for_filename(n_ratio),
          fmt_ratio_for_filename(m_ratio)
        )
      )
      write_selected_csv(selected_idx, out_path_grid)

      # store metrics for later averaging per SNR
      loo_mat[gg, rr] <- loo_final
      f1_mat[gg, rr]  <- f1

      # update best-by-LOO
      if (is.finite(loo_final) && loo_final < best_loo) {
        best_loo <- loo_final
        best_selected <- selected_idx
        best_n <- n_ratio
        best_m <- m_ratio
      }

      counter <- counter + 1
      setTxtProgressBar(pb, counter)
    }

    # save best per-seed selection (min LOO across 9 combos)
    best_out_path <- file.path(
      best_out_dir,
      sprintf(
        "adammars_selected_corr%g_snr%g_permute%d_oracle%d_rep%d.csv",
        corr, snr, permute, oracle, rep
      )
    )
    write_selected_csv(best_selected, best_out_path)

    cat(sprintf(
      "\nSNR=%g rep=%d best (n_ratio=%.2f, m_ratio=%.2f): LOO=%.6f | selected=%d | saved=%s\n",
      snr, rep, best_n, best_m, best_loo, length(best_selected), best_out_path
    ))
  }

  # compute per-grid averages across reps, write per-SNR grid summary CSV
  for (gg in seq_len(nrow(grid))) {
    grid_summary$avg_loo_final[gg] <- mean(loo_mat[gg, ], na.rm = TRUE)
    grid_summary$avg_f1[gg]        <- mean(f1_mat[gg, ],  na.rm = TRUE)
  }

  grid_csv_path <- file.path(
    grid_out_dir,
    sprintf("grid_search_snr%g_avg_loo_and_f1.csv", snr)
  )
  write.csv(grid_summary, grid_csv_path, row.names = FALSE)

  cat(sprintf("\nSaved per-SNR grid summary: %s\n", grid_csv_path))
}

close(pb)
cat("\nDONE.\n")
