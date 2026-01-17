
# print(max(50/((1-n_ratio)*m_ratio*m_ratio), 200/((1-n_ratio)*m_ratio)))

# --- set project root -------------------------------------------------
root <- "C:/Users/95815/Desktop/research/adamp"

# --- load AdaMP code --------------------------------------------------
source(file.path(root, "src", "R", "adamp", "adamp_main.R"))
source(file.path(root, "src", "R", "adamp", "fit_functions.R"))
source(file.path(root, "src", "R", "adamp", "simulation_functions.R"))

# --- fixed experiment setup ------------------------------------------
N  <- 200
M  <- 500
N1 <- 50

max_iter <- 5
permute  <- 0L

func_name <- "mr"
fit_funcs <- list(
  linear = linear_reg,
  mr     = mars_reg,
  sp     = spam_reg
)
fit_func <- fit_funcs[[func_name]]

delta <- 0.8

# tuning grid
n_ratio_list <- c(0.3, 0.35, 0.4, 0.45, 0.5)
m_ratio_list <- c(0.1, 0.12, 0.14, 0.16, 0.18)

# tuning scenario
snr  <- 5
corr <- 0.9

# 5 seeds / reps
rep_list <- 0:4

# true signal set: first 10 features (1..10)
signal_set <- 1:10

# --- output dirs ------------------------------------------------------
out_dir <- file.path(root, "temp_results", "hyperparam_tuning",
                     sprintf("nonlinear_additive_corr%g_snr%g_permute%d", corr, snr, permute))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(out_dir, "grid_search_avg_loo_and_f1.csv")

# --- helpers ----------------------------------------------------------
calc_K_one <- function(n_ratio, m_ratio) {
  denom1 <- (1 - n_ratio) * (m_ratio^2)
  denom2 <- (1 - n_ratio) * m_ratio
  if (denom1 <= 0 || denom2 <= 0) stop("Invalid (n_ratio, m_ratio)")
  k1 <- 50  / denom1
  k2 <- 200 / denom2
  as.integer(ceiling(max(k1, k2)))
}

fmt_ratio <- function(x, digits = 2) {
  gsub("\\.", "p", formatC(x, format = "f", digits = digits))
}

extract_final_loo <- function(res, max_iter) {
  key_last <- as.character(max_iter - 1L)
  if (!is.null(res[[key_last]]) && !is.null(res[[key_last]]$loo)) return(res[[key_last]]$loo)
  last_obj <- res[[length(res)]]
  if (!is.null(last_obj$loo)) return(last_obj$loo)
  stop("Could not extract final-epoch LOO from res.")
}

# selection rule
extract_selected_features <- function(res, max_iter, delta) {
  stopifnot(length(res) >= 1L)
  key_names  <- sort(names(res))
  num_epochs <- length(key_names)

  if (num_epochs == 1L) {
    choose_key_name <- key_names[1L]
  } else if (num_epochs == max_iter) {
    choose_key_name <- key_names[num_epochs]
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

run_one_rep <- function(n_ratio, m_ratio, rep_id,
                        root, corr, snr, permute, N, M, N1, fit_func, delta, max_iter, out_dir, signal_set) {

  data_path <- file.path(
    root, "data", "simulation",
    sprintf("nonlinear_additive_corr%g_snr%g_permute%d_rep%d.csv",
            corr, snr, permute, rep_id)
  )
  if (!file.exists(data_path)) stop("Data file not found: ", data_path)

  df <- read.csv(data_path)

  X <- as.matrix(df[, !(names(df) %in% c("Y"))])
  Y <- scale(df$Y)

  # placeholder
  sim <- SimuFriedmanAdditive(N, M, N1, snr = snr, seed = 42, corr = corr)
  X1 <- sim$X1
  Y1 <- sim$Y1

  # seed controls AdaMP randomness
  set.seed(rep_id)

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

  # save per-seed result
  fn <- sprintf("res_n%s_m%s_rep%d.rds", fmt_ratio(n_ratio), fmt_ratio(m_ratio), rep_id)
  saveRDS(res, file.path(out_dir, fn))

  loo_final <- extract_final_loo(res, max_iter)

  selected_idx <- extract_selected_features(res, max_iter, delta)
  f1 <- f1_from_selected(selected_idx, signal_set)

  list(loo_final = loo_final, f1 = f1, selected_idx = selected_idx)
}

# --- grid search ------------------------------------------------------
grid <- expand.grid(
  n_ratio = n_ratio_list,
  m_ratio = m_ratio_list,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

results <- data.frame(
  n_ratio = numeric(0),
  m_ratio = numeric(0),
  avg_loo_final = numeric(0),
  avg_f1 = numeric(0),
  stringsAsFactors = FALSE
)

for (ii in seq_len(nrow(grid))) {
  n_ratio <- grid$n_ratio[ii]
  m_ratio <- grid$m_ratio[ii]

  cat(sprintf("\n[%d/%d] n_ratio=%.2f, m_ratio=%.2f\n", ii, nrow(grid), n_ratio, m_ratio))

  loo_vec <- numeric(length(rep_list))
  f1_vec  <- numeric(length(rep_list))

  for (jj in seq_along(rep_list)) {
    rep_id <- rep_list[jj]
    cat(sprintf("  - rep %d ... ", rep_id))

    out <- run_one_rep(
      n_ratio = n_ratio, m_ratio = m_ratio, rep_id = rep_id,
      root = root, corr = corr, snr = snr, permute = permute,
      N = N, M = M, N1 = N1,
      fit_func = fit_func, delta = delta, max_iter = max_iter,
      out_dir = out_dir, signal_set = signal_set
    )

    loo_vec[jj] <- out$loo_final
    f1_vec[jj]  <- out$f1

    cat(sprintf("final LOO = %.6f | F1 = %.4f | selected = %d\n",
                loo_vec[jj], f1_vec[jj], length(out$selected_idx)))
  }

  avg_loo <- mean(loo_vec, na.rm = TRUE)
  avg_f1  <- mean(f1_vec,  na.rm = TRUE)

  results <- rbind(
    results,
    data.frame(
      n_ratio = n_ratio,
      m_ratio = m_ratio,
      avg_loo_final = avg_loo,
      avg_f1 = avg_f1,
      stringsAsFactors = FALSE
    )
  )

  # write after each grid point
  write.csv(results, csv_path, row.names = FALSE)

  cat(sprintf("  => avg final-epoch LOO = %.6f | avg F1 = %.4f | saved: %s\n",
              avg_loo, avg_f1, csv_path))
}

cat("\nDONE. Final CSV saved at:\n", csv_path, "\n", sep = "")
