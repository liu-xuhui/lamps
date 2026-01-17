# --- set project root -------------------------------------------------
root <- "C:/Users/95815/Desktop/research/adamp"

# --- load AdaMP code --------------------------------------------------
source(file.path(root, "src", "R", "adamp", "adamp_main.R"))
source(file.path(root, "src", "R", "adamp", "fit_functions.R"))
source(file.path(root, "src", "R", "adamp", "simulation_functions.R"))

N  <- 200
M  <- 500
N1 <- 50

max_iter <- 5

permute <- 0L   # use the same name everywhere

func_name <- "mr"
fit_funcs <- list(
  linear = linear_reg,
  mr     = mars_reg,
  sp = spam_reg
)
fit_func <- fit_funcs[[func_name]]

n_ratio <- 0.5
m_ratio <- 0.1
delta   <- 0.8
K       <- rep(10000, max_iter)

number_signals <- 10


out_dir <- file.path(root, "results", "other")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

rep <- 3
snr <- 5
corr <- 0.9

data_path <- file.path(
root, "data", "simulation",
sprintf("nonlinear_additive_corr%g_snr%g_permute%d_rep%d.csv",
        corr, snr, permute, rep)
)
df <- read.csv(data_path)

# predictors and response
X <- as.matrix(df[ , !(names(df) %in% c("Y"))])
#X <- scale(X)
Y <- df$Y
Y <- scale(Y)

sim <- SimuFriedmanAdditive(N, M, N1, snr = 5, seed = 42, corr = corr)
# useless
X1 <- sim$X1
Y1 <- sim$Y1

set.seed(3)

res <- indept_weight_sample_epochtuned(
    X, Y, X1, Y1,
    n_ratio = n_ratio,
    m_ratio = m_ratio,
    K       = K,
    fit_func = fit_func,
    delta    = delta,
    max_iter = max_iter
)



# Choose epoch
# stopifnot(length(res) >= 1L)
# key_names  <- sort(names(res))
# num_epochs <- length(key_names)

# if (num_epochs == 1L) {
# choose_key_name <- key_names[1L]
# } else if (num_epochs == max_iter) {
# choose_key_name <- key_names[num_epochs]
# } else {
# choose_key_name <- key_names[num_epochs - 1L]
# }

# prob_F_use <- res[[choose_key_name]]$prob_F


# if (is.null(prob_F_use)) stop("Selected epoch has no prob_F stored.")

# for (oracle in oracle_list) {

#     if (oracle == 0) {
#         nonlinear_select1 <- which(prob_F_use >= delta * 0.5)
#     } else {
#         nonlinear_select1 <- order(prob_F_use, decreasing = TRUE)[1:number_signals]
#     }

#     out_path <- file.path(
#         out_dir,
#         sprintf(
#         "selected01M_vars_add_corr%g_oracle%d_snr%g_permute%d_rep%d.csv",
#         corr, oracle, snr, permute, rep
#         )one_linear_corr{corr}_experiment_res
#     )

#     write.csv(data.frame(var = nonlinear_select1),
#                 out_path, row.names = FALSE)
# }

stopifnot(length(res) >= 1L)

# ---- sort epoch keys robustly (handles "0","1","2",...) ----
key_names <- names(res)
key_int   <- suppressWarnings(as.integer(key_names))
ord       <- order(key_int, na.last = TRUE)
key_names <- key_names[ord]

num_epochs <- length(key_names)

# ---- choose the epoch the same way you do ----
if (num_epochs == 1L) {
  choose_key_name <- key_names[1L]
} else if (num_epochs == max_iter) {
  choose_key_name <- key_names[num_epochs]
} else {
  choose_key_name <- key_names[num_epochs - 1L]
}

prob_F_use <- res[[choose_key_name]]$prob_F

nonlinear_select1 <- which(prob_F_use >= delta * 0.5)

stopifnot(!is.null(prob_F_use))

# ---- determine how many epochs to save (up to the epoch of prob_F_use) ----
choose_pos <- match(choose_key_name, key_names)
stopifnot(!is.na(choose_pos))
keys_to_save <- key_names[seq_len(choose_pos)]

# ---- build a M x E matrix: each column is prob_F for one epoch, in order ----
M_out <- length(prob_F_use)
prob_mat <- sapply(keys_to_save, function(k) {
  pf <- res[[k]]$prob_F
  if (is.null(pf)) rep(NA_real_, M_out) else as.numeric(pf)
})
if (is.vector(prob_mat)) prob_mat <- matrix(prob_mat, ncol = 1L)  # if only 1 epoch

colnames(prob_mat) <- paste0("epoch_", seq_len(ncol(prob_mat)))

# optional: include feature index as first column
out_df <- data.frame(feature = seq_len(M_out), prob_mat, check.names = FALSE)

# ---- save to CSV under out_dir ----
out_path <- file.path(
  out_dir,
  sprintf(
    "prob_F_epochs_0510_corr%g_snr%g_permute%d_rep%d_%s.csv",
    corr, snr, permute, rep, func_name
  )
)

write.csv(out_df, out_path, row.names = FALSE)
cat("Saved prob_F CSV to:", out_path, "\n")


print(num_epochs)
# prob_F_use1 <- res[[1]]$prob_F
# nonlinear_select3 <- which(prob_F_use1 >= delta * 0.5)
# print(nonlinear_select3)
# print(res[[1]]$loo)


