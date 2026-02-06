in_file <- file.path("data", "rosmap", "rosmap_200.csv")
if (!file.exists(in_file)) {
  stop("ROSMAP file not found: ", in_file)
}

df_raw <- read.csv(in_file, check.names = FALSE)

## Basic structure checks
if (ncol(df_raw) < 4) {
  stop("ROSMAP file must have at least 4 columns (index, >=1 X, useless, y).")
}

## Identify columns by position
idx_col        <- 1L
useless_col    <- ncol(df_raw) - 1L
y_col          <- ncol(df_raw)

## Response (last column)
Y <- df_raw[[y_col]]

## Predictors: drop first, second-last, last
X_df <- df_raw[, -c(idx_col, useless_col, y_col), drop = FALSE]

## Convert to numeric matrix
X <- as.matrix(X_df)
storage.mode(X) <- "double"

stopifnot(nrow(X) == length(Y))

## Output directory
out_dir <- file.path("data", "rosmap")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

## Parameters
n_splits   <- 10
train_frac <- 0.8
n          <- nrow(X)
n_train    <- floor(train_frac * n)

## Clean predictor names for Python
colnames(X) <- paste0("X", seq_len(ncol(X)))

## Combine into one data frame
full_data <- data.frame(
  Y = Y,
  X
)

## Generate splits (random 80/20)
for (i in seq_len(n_splits)) {

  set.seed(110 + i)

  train_idx <- sample(seq_len(n), size = n_train, replace = FALSE)
  test_idx  <- setdiff(seq_len(n), train_idx)

  train_data <- full_data[train_idx, ]
  test_data  <- full_data[test_idx, ]

  train_file <- file.path(out_dir, sprintf("rosmap_train_split_%02d.csv", i))
  test_file  <- file.path(out_dir, sprintf("rosmap_test_split_%02d.csv", i))

  write.csv(train_data, train_file, row.names = FALSE)
  write.csv(test_data,  test_file,  row.names = FALSE)
}

cat("ROSMAP random train-test splits generated successfully.\n")
cat("Input file:", in_file, "\n")
cat("n =", n, "\n")
cat("p (predictors) =", ncol(X), "\n")
cat("Saved under:", out_dir, "\n")