## ----------------------------------------------------------------------------
##
## Script name: Fig1_hyndman_paper.R
## Purpose of script: Reproduce Rob J. Hyndman fig.1 of the paper
##                    https://arxiv.org/abs/2008.00444
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

library(reshape)
library(ggplot2)
library(testit)

DNAME <- "M3"
data_origin <- utils.load_dataset(DNAME)
SUBSET <- "MONTHLY"
data <- data_origin[as.vector(unlist(lapply(data_origin, function(s) s$period == SUBSET)))]
FORECASTING_HORIZON <- unique(as.vector(unlist(lapply(data, function(x) x$h))))
assert(length(FORECASTING_HORIZON)==1)

metrics <- NULL
metrics$glinear <- list()
metrics$gjoint <- list()

MIN_LAG <- 1
MAX_LAG <- 50
pb = txtProgressBar(
  min = MIN_LAG,
  max = MAX_LAG,
  initial = 0,
  style = 3,
  width = 100
)
for (LAG in MIN_LAG:MAX_LAG) {
  collection <- utils.build_lagged_dataset(data,
                                           LAG,
                                           FORECASTING_HORIZON,
                                           verbose = FALSE)
  lm <- LinearModel$new(collection$X_train, collection$y_train)
  y_hat <- lm$predict(collection$X_test, h = FORECASTING_HORIZON)
  metrics$glinear[[LAG-MIN_LAG+1]] <- mean(metrics.mase$compute(y_hat$pf, collection$Y_test))
  
  jm <- JointModel$new(collection$X_train, collection$y_train)
  y_hat <- jm$predict(collection$X_test, h = FORECASTING_HORIZON)
  metrics$gjoint[[LAG-MIN_LAG+1]] <- mean(metrics.mase$compute(y_hat$pf, collection$Y_test))
  
  local <- NULL
  N <- length(collection$data)
  local$auto.arima$pf <-
    matrix(nrow = N, ncol = FORECASTING_HORIZON)
  local$ets$pf <- matrix(nrow = N, ncol = FORECASTING_HORIZON)
  local$theta$pf <- matrix(nrow = N, ncol = FORECASTING_HORIZON)
  series_id <- as.vector(unlist(lapply(collection$data, function(s) s$ID)))
  current_idx = 1
  for (i in series_id) {
    local$auto.arima$pf[current_idx, ] <-
      data_origin[[i]]$lforecast$auto_arima$mean
    local$ets$pf[current_idx, ] <-
      data_origin[[i]]$lforecast$ets$mean
    local$theta$pf[current_idx, ] <-
      data_origin[[i]]$lforecast$theta$mean
    current_idx = current_idx + 1
  }
  metrics$auto.arima[[LAG-MIN_LAG+1]] <-
    mean(rowMeans(abs(
      local$auto.arima$pf - collection$Y_test
    )))
  metrics$ets[[LAG-MIN_LAG+1]] <-
    mean(rowMeans(abs(local$ets$pf - collection$Y_test)))
  metrics$theta[[LAG-MIN_LAG+1]] <-
    mean(rowMeans(abs(local$theta$pf - collection$Y_test)))
  setTxtProgressBar(pb, LAG)
}

close(pb)

metrics$tab <- matrix(nrow = length(attributes(metrics)$names), ncol = LAG-MIN_LAG+1)
for (i in 1:length(attributes(metrics)$names)-1) {
  metrics$tab[i,] <- as.vector(unlist(metrics[i]))
}
metrics$tab <- t(metrics$tab)
metrics$tab <- cbind(metrics$tab, c(MIN_LAG:MAX_LAG))
colnames(metrics$tab) <- c("Global Linear AR", "Joint", "ARIMA", "ETS", "THETA", "LAG")
metrics$tab <- melt(as.data.frame(metrics$tab), id = c("LAG"))
colnames(metrics$tab) <- c("LAG", "Model", "MASE")
metrics$tab

p <-
  ggplot(data = metrics$tab, aes(x = LAG, y = MASE, group = Model)) +
  geom_line(aes(color = Model, linetype = Model)) +
  scale_color_manual(values = c("#000000", "#ff0000", "#948b85", "#eb7d34", "#2d83c4")) +
  theme_classic() +
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dotted", "twodash"))
p <- p + xlim(10,50) + ylim(0.80,1.25)
p
