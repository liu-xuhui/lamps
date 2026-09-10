library(SAM)

## -------------------------------------------------------------------------
## Settings
## -------------------------------------------------------------------------
seed <- 118L
n_splits <- 10L
max_features <- 20L
n_lambdas <- 200L

data_dir <- file.path("data", "rosmap")
out_dir <- file.path("results", "rosmap")
out_file <- file.path(out_dir, "rosmap_spam.csv")


## Choose the strongest lambda whose model selects exactly target features.
## If no exact path point exists, choose the closest support size. On a tie,
## prefer at least target features and then the stronger regularization.
choose_lambda_index <- function(lambda_vec, func_norm, target) {
  if (!is.matrix(func_norm)) {
    stop("'func_norm' must be a matrix with features in rows and lambdas in columns.")
  }
  if (ncol(func_norm) != length(lambda_vec)) {
    stop("The number of lambda values does not match the columns of 'func_norm'.")
  }
  if (anyNA(func_norm) || anyNA(lambda_vec)) {
    stop("SPAM returned missing functional norms or lambda values.")
  }

  nonzero_counts <- colSums(func_norm != 0)
  exact <- which(nonzero_counts == target)

  if (length(exact) > 0L) {
    index <- exact[which.max(lambda_vec[exact])]
    return(list(
      index = index,
      exact = TRUE,
      n_selected = nonzero_counts[index]
    ))
  }

  distance <- abs(nonzero_counts - target)
  closest <- which(distance == min(distance))

  ## Match the deterministic fallback used by run_lasso_new.py.
  at_least_target <- closest[nonzero_counts[closest] >= target]
  if (length(at_least_target) > 0L) {
    closest <- at_least_target
  }
  index <- closest[which.max(lambda_vec[closest])]

  list(
    index = index,
    exact = FALSE,
    n_selected = nonzero_counts[index]
  )
}


## Produce a JSON array using base R so Python can parse the feature set
## without requiring an additional R package such as jsonlite.
format_index_set <- function(indices) {
  paste0("[", paste(as.integer(indices), collapse = ", "), "]")
}


if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

set.seed(seed)
selected_sets <- vector("list", n_splits * max_features)
record_index <- 1L

for (split_id in seq_len(n_splits)) {
  train_file <- file.path(
    data_dir,
    sprintf("rosmap_train_split_%02d.csv", split_id)
  )

  if (!file.exists(train_file)) {
    stop("Training file not found: ", train_file)
  }

  train_df <- read.csv(train_file, check.names = FALSE)
  if (!("Y" %in% names(train_df))) {
    stop("Column 'Y' not found in: ", train_file)
  }

  Y_train <- train_df$Y
  X_train <- as.matrix(
    train_df[, setdiff(names(train_df), "Y"), drop = FALSE]
  )
  storage.mode(X_train) <- "double"

  if (nrow(X_train) != length(Y_train)) {
    stop("Predictor and response row counts differ in: ", train_file)
  }
  if (any(!is.finite(X_train)) || any(!is.finite(Y_train))) {
    stop("Non-finite data found in: ", train_file)
  }

  ## Reset the seed per split so results do not depend on loop execution order.
  set.seed(seed + split_id - 1L)
  fit <- samQL(X_train, Y_train, nlambda = n_lambdas)

  if (is.null(fit$lambda) || is.null(fit$func_norm)) {
    stop("SPAM fit is missing 'lambda' or 'func_norm' for split ", split_id, ".")
  }

  split_summary <- character(max_features)

  for (target in seq_len(max_features)) {
    chosen <- choose_lambda_index(fit$lambda, fit$func_norm, target)
    lambda_index <- chosen$index
    norms <- fit$func_norm[, lambda_index]

    ## Convert R's one-based feature indices to zero-based Python indices.
    selected_indices <- which(norms != 0) - 1L
    if (length(selected_indices) != chosen$n_selected) {
      stop(
        sprintf(
          "Selected-count mismatch for split %02d, K=%d.",
          split_id,
          target
        )
      )
    }

    if (!chosen$exact) {
      warning(
        sprintf(
          paste0(
            "Split %02d: no lambda produced exactly %d selected features; ",
            "using lambda=%.12g, which selected %d features."
          ),
          split_id,
          target,
          fit$lambda[lambda_index],
          chosen$n_selected
        ),
        call. = FALSE,
        immediate. = TRUE
      )
    }

    selected_sets[[record_index]] <- data.frame(
      split = sprintf("split_%02d", split_id),
      K = target,
      lambda = fit$lambda[lambda_index],
      n_selected = chosen$n_selected,
      selected_features = format_index_set(selected_indices),
      stringsAsFactors = FALSE
    )
    record_index <- record_index + 1L

    split_summary[target] <- sprintf(
      "k=%d:lambda=%.6g/n=%d",
      target,
      fit$lambda[lambda_index],
      chosen$n_selected
    )
  }

  message(sprintf("split %02d: %s", split_id, paste(split_summary, collapse = ", ")))
}

output <- do.call(rbind, selected_sets)
write.csv(output, out_file, row.names = FALSE, quote = TRUE)

cat("SPAM feature-selection results saved to:\n", out_file, "\n", sep = "")
