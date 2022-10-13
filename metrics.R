## ----------------------------------------------------------------------------
##
## Script name: metrics
## Purpose of script: given actual and forecast compute errors, and summaries
##
## Author: Nicolo' Rubattu
##
## Date Created: 12-10-2022
## Copyright (c) Nicolo' Rubattu, 2022
## Email: nicolo.rubattu@idsia.ch
## ----------------------------------------------------------------------------


# Mean Absolute Error (MAE) ---------------------------------------------------
metrics.mae <- NULL
metrics.mae$name <- "SMAPE"
metrics.mae$compute <- function(y_hat, y, mase_scal_vec) {
  err <- mean(abs(y_hat - y))
  err
}

# Mean Absolute Scaled Error (MASE) -------------------------------------------
# Generic Error for MASE but any norm, 1 is MASE, 2 is MSSE, ...
metrics.mase <- NULL
metrics.mase$name <- "MASE"
metrics.mase$compute <- function(y_hat, y) {
  err <- rowMeans(abs(y_hat - y))
  err
}

# Symmetric Mean Absolute Percentage Error (sMAPE) ----------------------------
metrics.smape <- NULL
metrics.smape$name <- "SMAPE"
metrics.smape$compute <- function(y_hat, y) {
  denom <- 0.5 * (abs(y_hat) + abs(y))
  err <- abs(y_hat -  y) / denom
  err[is.infinite(err)] <- 0
  err[abs(denom) < 1e-8] <- 0
  err <- 100 * rowMeans(err)
  err
}

###############################################################################
# Compute metrics given predicted and actual values ---------------------------
metrics.compute_metrics <- function(predicted, actual, metrics, h) {
  tables <- list()
  metrics_idx <- 1
  union <- c(predicted$global, predicted$local)
  models_name <-
    attributes(lapply(union, function(m) {
      m$name
    }))$names
  for (m in metrics) {
    print(paste0("Computing ", m$name, "..."))
    tab <- matrix(nrow = 1, ncol = length(union))
    model_i = 1
    for (model_prediction in union) {
      tab[1, model_i] <-
        mean(m$compute(model_prediction$pf, actual))
      model_i = model_i + 1
    }
    colnames(tab) <- models_name
    rownames(tab) <- m$name
    tables[[metrics_idx]] <- as.data.frame(tab)
    metrics_idx = metrics_idx + 1
  }
  metrics_table <- do.call(rbind, tables)
  metrics_table
}
