library(earth)
library(stringr)


oracle <- 0
features_to_choose <- 30
n_splits <- 10


## Paths
data_dir <- file.path("data", "rosmap")
out_dir  <- file.path("results", "rosmap")

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
  set.seed(110 + i)

  fit <- earth(x = X_train, y = Y_train)
  # Get selected variable indices
  term_names <- rownames(fit$dirs[fit$selected.terms, ])

  # Remove intercept
  term_names <- term_names[term_names != "(Intercept)"]

  # Extract unique variable indices using regex
  selected_vars <- unique(as.integer(sub(".*X(\\d+).*", "\\1", term_names))) - 1L

  selected_mat_0idx[1:length(selected_vars), i] <- selected_vars

  print(i)

  
}

out_file <- file.path(out_dir,"rosmap_mars_nonoracle.csv")

write.csv(selected_mat_0idx, out_file, row.names = TRUE)

cat("Saved selected feature indices to:\n", out_file, "\n", sep = "")




