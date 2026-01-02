
library(RhpcBLASctl)

blas_set_num_threads(1)
## -------------------------------------------------------------------
## Load AdaMP code
## -------------------------------------------------------------------
root <- "C:/Users/95815/Desktop/research/adamp"

source(file.path(root, "src", "R", "adamp", "adamp_main.R"))
source(file.path(root, "src", "R", "adamp", "fit_functions.R"))
source(file.path(root, "src", "R", "adamp", "simulation_functions.R"))  # optional

## -------------------------------------------------------------------
## Settings
## -------------------------------------------------------------------
N  <- 200
M  <- 50
N1 <- 50

max_iter <- 5

permute <- 0L   # use the same name everywhere

func_name <- "sp"
fit_funcs <- list(
  linear = linear_reg,
  mr     = mars_reg,
  sp = spam_reg,
  mr2 = mars_reg2
)
fit_func <- fit_funcs[[func_name]]

n_ratio <- 0.4
m_ratio <- 0.12
delta   <- 0.8
K       <- rep(5787, max_iter)  # one K per epoch

number_signals <- 10

snr_list    <- c(4, 6, 8, 10)
num_rep     <- 30
p           <- 50
corr_list   <- c(0, 0.5, 0.9)
oracle_list <- c(0, 1)

out_dir <- file.path(root, "temp_results", "nonlinear", "nonadditive")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

## -------------------------------------------------------------------
## Main loop
## -------------------------------------------------------------------
total_runs <- length(corr_list) * length(snr_list) * num_rep
pb <- txtProgressBar(min = 0, max = total_runs, style = 3)
counter <- 0

for (corr in corr_list) {
  for (snr in snr_list) {

    for (rep in 0:(num_rep - 1)) {

      data_path <- file.path(
        root, "data", "simulation",
        sprintf("nonlinear_nonadditive_corr%g_snr%g_permute%d_rep%d.csv",
                corr, snr, permute, rep)
      )
      df <- read.csv(data_path)

      # predictors and response
      X <- as.matrix(df[ , !(names(df) %in% c("Y"))])
      Y <- df$Y
      Y <- scale(Y)

      # useless
      sim <- SimuFriedmanAdditive(N, M, N1, snr = 5, seed = 42, corr = 0)
      X1 <- sim$X1
      Y1 <- sim$Y1

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

      for (oracle in oracle_list) {

        if (oracle == 0) {
          nonlinear_select1 <- which(prob_F_use >= delta * 0.5)
        } else {
          nonlinear_select1 <- order(prob_F_use, decreasing = TRUE)[1:number_signals]
        }

        if (func_name == "mr2") {
          out_path <- file.path(
            out_dir,
            sprintf(
              "adammars_selected_corr%g_snr%g_permute%d_oracle%d_rep%d.csv",
              corr, snr, permute, oracle, rep
            )
          )
        }
        else {
          out_path <- file.path(
            out_dir,
            sprintf(
              "adamspam_selected_corr%g_snr%g_permute%d_oracle%d_rep%d.csv",
              corr, snr, permute, oracle, rep
            )
          )
        }
        

        write.csv(data.frame(var = nonlinear_select1),
                  out_path, row.names = FALSE)
      }

      counter <- counter + 1
      setTxtProgressBar(pb, counter)
    }
  }
}
close(pb)
