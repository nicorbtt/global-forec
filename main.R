## ----------------------------------------------------------------------------
##
## Script name: main
## Purpose of script: entry point for the experiments
##
## Author: Nicolo' Rubattu
##
## Date Created: 12-10-2022
## Copyright (c) Nicolo' Rubattu, 2022
## Email: nicolo.rubattu@idsia.ch
## ----------------------------------------------------------------------------


source("models.R")
source("metrics.R")
source("utils.R")
source("plotting.R")

DNAME <- "M1"
data_origin <- utils.load_dataset(DNAME)
SUBSET <- "MONTHLY"
data <-
  data_origin[as.vector(unlist(lapply(data_origin, function(s)
    s$period == SUBSET)))]
FORECASTING_HORIZON <- data[[1]]$h
LAG <- 25
collection <- utils.build_lagged_dataset(data,
                                         LAG,
                                         FORECASTING_HORIZON)
rm(data)

y_hat <- NULL
### GLOBAL MODELS -------------------------------------------------------------
y_hat$global <- NULL
lm <-
  LinearModel$new(collection$X_train, collection$y_train) # LINEAR
y_hat$global$Linear <-
  lm$predict(collection$X_test, h = FORECASTING_HORIZON)
#
jm <- JointModel$new(collection$X_train, collection$y_train) # JOINT
y_hat$global$Joint <-
  jm$predict(collection$X_test, h = FORECASTING_HORIZON)
#
# rf <- RandomForest$new(collection$X_train, collection$y_train) # RANDOM FOREST
# y_hat$global$RandomForest <- rf$predict(collection$X_test, h = FORECASTING_HORIZON)
#
# xgb <- XGB$new(collection$X_train, collection$y_train) # XGB
# y_hat$global$XGB <- xgb$predict(collection$X_test, h = FORECASTING_HORIZON)
#
# deepNetAR <- DeepNetAR$new(collection$X_train, collection$y_train) # DEEPNET AR
# y_hat$global$DeepNetAR <-
#   deepNetAR$predict(collection$X_test, h = FORECASTING_HORIZON)

### METRICS COMPUTATION -------------------------------------------------------
y_hat$local <- NULL
N <- length(collection$data)
y_hat$local$auto.arima$pf <-
  matrix(nrow = N, ncol = FORECASTING_HORIZON)
y_hat$local$ets$pf <- matrix(nrow = N, ncol = FORECASTING_HORIZON)
y_hat$local$theta$pf <- matrix(nrow = N, ncol = FORECASTING_HORIZON)
series_id <-
  as.vector(unlist(lapply(collection$data, function(s)
    s$ID)))
current_idx = 1
for (i in series_id) {
  y_hat$local$auto.arima$pf[current_idx,] <-
    as.numeric(data_origin[[i]]$lforecast$auto_arima$mean)
  y_hat$local$ets$pf[current_idx,] <-
    as.numeric(data_origin[[i]]$lforecast$ets$mean)
  y_hat$local$theta$pf[current_idx,] <-
    as.numeric(data_origin[[i]]$lforecast$theta$mean)
  current_idx = current_idx + 1
}

metrics_table <- metrics.compute_metrics(
  predicted = y_hat,
  actual = collection$Y_test,
  metrics = list(metrics.mase, metrics.smape),
  h = FORECASTING_HORIZON
)
metrics_table
