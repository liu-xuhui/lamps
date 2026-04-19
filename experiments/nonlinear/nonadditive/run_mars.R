library(earth)
library(stringr)

# ----- Configuration -----

snr_list <- c(4,6,8,10)
num_rep <- 30
p = 50
corr_list = c(0,0.5,0.9)
permute = as.integer(Sys.getenv("ADAMP_PERMUTE", "0"))
oracle = 0 # mars can only have oracle = 0

out_dir <- "temp_results/nonlinear/nonadditive/"


# ----- Main Loop -----
for (j in seq_along(corr_list)) {
  corr = corr_list[j]
  for (i in seq_along(snr_list)) {
    snr <- snr_list[i]
    f1_vec <- c()

    for (rep in 0:(num_rep - 1)) {
      # Read data
      data_path <- sprintf("data/simulation/nonlinear_nonadditive_corr%g_snr%g_permute%d_rep%d.csv", corr, snr, permute, rep)
      
      df <- read.csv(data_path)

      # Fit MARS model
      predictor_data <- as.matrix(df[ , !(names(df) %in% c("Y"))])  # all columns except Y
      response_vector <- df$Y

      # Fit MARS model using matrix interface
      fit <- earth(x = predictor_data, y = response_vector, degree = 2)

      # Get selected variable indices
      term_names <- rownames(fit$dirs[fit$selected.terms, ])

      # Remove intercept
      term_names <- term_names[term_names != "(Intercept)"]

      # Extract unique variable indices using regex
      selected_vars <- unique(as.integer(sub(".*X(\\d+).*", "\\1", term_names)))

      out_path <- file.path(out_dir, sprintf("mars_selected_corr%g_snr%g_permute%d_oracle%d_rep%d.csv",corr, snr, permute, oracle, rep))
      
      write.csv(data.frame(var = selected_vars), out_path, row.names = FALSE)

      print(rep)
    }
  }
}
