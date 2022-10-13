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
metrics.mase$compute <-
  function(y_hat, y, mase_scal_vec, norm = 1) {
    y_hat <- y_hat * mase_scal_vec
    y <- y * mase_scal_vec
    err <- mean(abs(y_hat - y) ** norm / mase_scal_vec ** norm)
    err
  }

# Symmetric Mean Absolute Percentage Error (sMAPE) ----------------------------
metrics.smape <- NULL
metrics.smape$name <- "SMAPE"
metrics.smape$compute <- function(y_hat, y, mase_scal_vec) {
  y_hat <- y_hat * mase_scal_vec
  y <- y * mase_scal_vec
  denom <- 0.5 * (abs(y_hat) + abs(y))
  err <- abs(y_hat -  y) / denom
  err[is.infinite(err)] <- 0
  err[abs(denom) < 1e-8] <- 0
  err <- 100 * mean(err)
  err
}

###############################################################################
# Compute metrics given predicted and actual values ---------------------------
metrics.compute_metrics <-
  function(predicted, actual, mase_scal, metrics, h) {
    tables <- list()
    metrics_idx <- 1
    union <- c(predicted$global, predicted$local)
    models_name <-
      attributes(lapply(union, function(m) {
        m$name
      }))$names
    for (m in metrics) {
      print(paste0("Computing ", m$name, "..."))
      tmp <- NULL
      tmp$name <- m$name
      tmp$tab <- matrix(nrow = h, ncol = length(union))
      model_i = 1
      for (model_prediction in union) {
        for (h_ in 1:h) {
          tmp$tab[h_, model_i] <-
            m$compute(model_prediction$pf[, h_], actual[, h_], mase_scal)
        }
        model_i = model_i + 1
      }
      rownames(tmp$tab) <- paste0("h=", 1:h)
      colnames(tmp$tab) <- models_name
      tmp$tab <- as.table(tmp$tab)
      tables[[metrics_idx]] <- tmp
      metrics_idx = metrics_idx + 1
    }
    tables
  }
