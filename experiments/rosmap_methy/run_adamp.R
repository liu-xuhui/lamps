library(RhpcBLASctl)

blas_set_num_threads(1)
## -------------------------------------------------------------------
## Load AdaMP code
## -------------------------------------------------------------------
library(here)
root <- here::here()

source(file.path(root, "src", "R", "adamp", "adamp_main.R"))
source(file.path(root, "src", "R", "adamp", "fit_functions.R"))

max_iter <- 5
n_ratio <- 0.4
m_ratio <- 0.12
delta   <- 0.8
K       <- rep(5787, max_iter)
features_to_choose <- 50

n_splits <- 10

oracle <- 0    # 0 or 1
delta <- 0.8

## Choose base model: "mr" (MARS) or "sp" (SPAM)
func_name <- "mr"

fit_funcs <- list(
  linear = linear_reg,
  mr     = mars_reg,
  sp     = spam_reg
)
fit_func <- fit_funcs[[func_name]]
stopifnot(!is.null(fit_func))

## Paths
data_dir <- file.path("data", "rosmap_methy")
out_dir  <- file.path("results", "rosmap_methy")

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

## Storage: 50 x 10 matrix
## Each column i holds top-50 selected feature indices for split i (0-indexed)
selected_mat_0idx <- matrix(NA_integer_, nrow = features_to_choose, ncol = n_splits)
colnames(selected_mat_0idx) <- sprintf("split_%02d", seq_len(n_splits))
rownames(selected_mat_0idx) <- paste0("rank_", seq_len(features_to_choose))

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
  set.seed(10000 + i + ifelse(func_name == "mr", 0, 1000))

  ## Run AdaMP
  res <- indept_weight_sample_epochtuned(
    X_train, Y_train, X_train, Y_train,
    n_ratio  = n_ratio,
    m_ratio  = m_ratio,
    K        = K,
    fit_func = fit_func,
    delta    = delta,
    max_iter = max_iter
  )

  ## Choose epoch key
  key_names  <- sort(names(res))
  num_epochs <- length(key_names)

  if (num_epochs == 1L) {
    choose_key_name <- key_names[1L]
  } else {
    choose_key_name <- key_names[num_epochs - 1L]
  }

  prob_F_use <- res[[choose_key_name]]$prob_F

  if (oracle == 1L) {
    ## Top features in decreasing importance (R index: 1..p)
    top_idx_1 <- order(prob_F_use, decreasing = TRUE)[seq_len(features_to_choose)]

    ## Convert to 0-index for Python
    top_idx_0 <- top_idx_1 - 1L

    selected_mat_0idx[, i] <- top_idx_0
  }
  else {
    nonlinear_select1 <- which(prob_F_use >= delta * 0.5)

    nonlinear_select0 <- nonlinear_select1 - 1L

    selected_mat_0idx[1:length(nonlinear_select0), i] <- nonlinear_select0

  }

  
}

## Save 50 x 10 CSV
## Make filename informative

# if (oracle == 1) {
#     out_file <- file.path(
#     out_dir,
#     sprintf(
#         "rosmap_adamp_%s.csv",
#         func_name
#     )
# )
# }
# else {
#     out_file <- file.path(
#     out_dir,
#     sprintf(
#         "rosmap_adamp_nonoracle_%s.csv",
#         func_name
#     )
# )
# }


# out_file <- file.path(
# out_dir,
# sprintf(
#     "rosmap_adamp_%s.csv",
#     func_name
#   )
# )

out_file <- file.path(
out_dir,
sprintf(
        "rosmap_adamp_nonoracle_%s.csv",
        func_name
    )
)

write.csv(selected_mat_0idx, out_file, row.names = TRUE)

cat("Saved selected feature indices to:\n", out_file, "\n", sep = "")