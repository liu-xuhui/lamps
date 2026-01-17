library(earth)
library(stringr)
library(SAM)
library(future)
library(future.apply)


linear_reg <- function(X, Y, X1) {
  # Fit linear regression without intercept
  model <- lm(Y ~ . - 1, data = as.data.frame(X))  
  
  # Predict on new data
  preds <- as.numeric(predict(model, newdata = as.data.frame(X1)))
  
  return(preds)
}

spam_reg <- function(X, Y, X1) {
  fit_best <- samQL(X, Y, lambda = c(0.01))
  preds <- as.numeric(predict(fit_best, newdata = X1)$values[,1])
  return(preds)
}

spam_reg_lambvar <- function(X, Y, X1) {
  fit_best <- samQL(X, Y, lambda = c(0.0001))
  preds <- as.numeric(predict(fit_best, newdata = X1)$values[,1])
  return(preds)
}

mars_reg <- function(X, Y, X1) {

  model <- earth::earth(
    x = X,
    y = Y,
    trace = 0     # suppress output
  )

  preds <- as.numeric(predict(model, newdata = X1))
  return(preds)
}

mars_reg2 <- function(X, Y, X1) {
  model <- earth::earth(
    x = X,
    y = Y,
    trace = 0,
    degree = 2
  )
  preds <- as.numeric(predict(model, newdata = X1))
  return(preds)
}
