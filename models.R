## ----------------------------------------------------------------------------
##
## Script name: models
## Purpose of script: implementation of global and local methods for time
##                    series forecasting
##
## Author: Nicolo' Rubattu
##
## Date Created: 12-10-2022
## Copyright (c) Nicolo' Rubattu, 2022
## Email: nicolo.rubattu@idsia.ch
## ----------------------------------------------------------------------------


library(R6)             # https://r6.r-lib.org/articles/Introduction.html
library(corpcor)
library(randomForest)
library(xgboost)
library(keras)
library(forecast)
library(pbapply)
library(parallel)

SEED = 42
set.seed(SEED)
VAL_SIZE = 0.20
PATIENCE = 20

################################## GLOBAL #####################################
# LINEAR REGRESSION MODEL------------------------------------------------------
LinearModel <- R6Class(
  classname = "LinearModel",
  private = list(fitted_model = NULL,
                 train_colnames = NULL),
  public = list(
    initialize = function(xtrain, ytrain) {
      colnames(xtrain) <- NULL
      lm_train <- data.frame(X = xtrain, Y = ytrain)
      if (length(xtrain[1, ]) == 1) {
        train_colnames <- "X"
      } else {
        train_colnames <- paste("X", 1:length(xtrain[1, ]), sep = ".")
      }
      formula_str <-
        paste0("Y ~ ", paste(train_colnames, collapse = " + "), " - 1")
      lm.fit <- lm(as.formula(formula_str), data = lm_train)
      private$fitted_model <- lm.fit
      private$train_colnames <- train_colnames
    },
    predict = function(xtest, h = 1) {
      xtest <- as.data.frame(xtest)
      pf  <- matrix(nrow = length(xtest[, 1]), ncol = h)
      for (i in 1:h) {
        if (i > 1) {
          if (length(xtest[1,])==1) {
            xtest <- data.frame(y_hat_h)
          } else {
            xtest <- cbind(xtest[2:ncol(xtest)], y_hat_h)
          }
        }
        colnames(xtest) <- private$train_colnames
        y_hat_h <- predict(private$fitted_model, newdata = xtest)
        pf[, i] <- y_hat_h
      }
      forecast <- NULL
      forecast$pf <- pf
      forecast
    }
  )
)

# JOINT MODEL------------------------------------------------------------------
JointModel <- R6Class(
  classname = "JointModel",
  private = list(
    S = NULL,
    inv_Sx = NULL,
    S_yx = NULL,
    mu_x = NULL,
    mu_y = NULL,
    var_y = NULL,
    cache = NULL,
    train_colnames = NULL
  ),
  public = list(
    initialize = function(xtrain, ytrain) {
      xy      = cbind(xtrain, ytrain)
      k       = ncol(xy)
      S       = cov.shrink(xy, verbose = FALSE)
      inv_Sx  = solve(S[1:(k - 1), 1:(k - 1)])
      S_yx    = t(S[k, 1:(k - 1)])
      mu_x    = colMeans(xtrain)
      mu_y    = mean(ytrain)
      var_y   = var(ytrain)
      cache   = S_yx  %*% inv_Sx
      
      private$S <- S
      private$inv_Sx <- inv_Sx
      private$S_yx <- S_yx
      private$mu_x <- mu_x
      private$mu_y <- mu_y
      private$var_y <- var_y
      private$cache <- cache
      private$train_colnames <- colnames(xtrain)
    },
    predict = function(xtest, h = 1) {
      var <- vector(length = h)
      pf  <- matrix(nrow = length(xtest[, 1]), ncol = h)
      for (i in 1:h) {
        if (i > 1) {
          xtest <- cbind(xtest[, 2:ncol(xtest)], y_hat_h)
          colnames(xtest) <- private$train_colnames
        }
        #VAR (#TODO fix variance)
        var[i] <-
          sum(var) + private$var_y - private$cache %*% t(private$S_yx)
        #MEAN
        L <- length(xtest[, 1])
        mu_x_vec <- t(matrix(unlist(rep(private$mu_x, L)),
                             nrow = length(xtest[1, ]),
                             ncol = L))
        x_mu_x_vec <- xtest - mu_x_vec
        y_hat_h <-
          as.vector(private$cache %*% t(x_mu_x_vec) + private$mu_y)
        pf[, i] <- y_hat_h
      }
      forecast <- NULL
      forecast$pf <- pf
      forecast$var <- var
      forecast
    }
  )
)

# RF REGRESSION MODEL----------------------------------------------------------
RandomForest <- R6Class(
  classname = "RandomForest",
  private = list(fitted_model = NULL,
                 train_colnames = NULL),
  public = list(
    initialize = function(xtrain, ytrain) {
      private$fitted_model <-
        randomForest(xtrain, ytrain, do.trace = TRUE)
      private$train_colnames <- colnames(xtrain)
    },
    predict = function(xtest, h = 1) {
      pf  <- matrix(nrow = length(xtest[, 1]), ncol = h)
      for (i in 1:h) {
        if (i > 1) {
          xtest <- cbind(xtest[2:ncol(xtest)], y_hat_h)
        }
        colnames(xtest) <- private$train_colnames
        y_hat_h <- predict(private$fitted_model, xtest)
        pf[, i] <- y_hat_h
      }
      forecast <- NULL
      forecast$pf <- pf
      forecast
    }
  )
)

# XGBOOST REGRESSION MODEL-----------------------------------------------------
XGB <- R6Class(
  classname = "XGB",
  private = list(fitted_model = NULL,
                 train_colnames = NULL),
  public = list(
    initialize = function(xtrain, ytrain) {
      ind_cv = sample(nrow(xtrain), floor(nrow(xtrain) * VAL_SIZE))
      xval = xtrain[ind_cv, , drop = FALSE]
      yval = ytrain[ind_cv]
      xtrain = xtrain[-ind_cv, , drop = FALSE]
      ytrain = ytrain[-ind_cv]
      sampind = sample(nrow(xtrain))
      xtrain = xtrain[sampind, , drop = FALSE]
      ytrain = ytrain[sampind]
      
      dtrain <- xgb.DMatrix(as.matrix(xtrain), label = ytrain)
      dtest <- xgb.DMatrix(as.matrix(xval), label = yval)
      watchlist <- list(train = dtrain, eval = dtest)
      xgb_model = xgb.train(
        params = list(
          booster = "gbtree",
          objective = "reg:squarederror",
          eta = 0.03,
          max_depth = 16,
          colsample_bytree = 0.9,
          subsample = 0.9,
          eval_metric = "mae"
        ),
        dtrain,
        nrounds = 1000,
        watchlist = watchlist,
        early_stopping_rounds = PATIENCE
      )
      private$fitted_model <- xgb_model
      private$train_colnames <- colnames(xtrain)
    },
    predict = function(xtest, h = 1) {
      pf  <- matrix(nrow = length(xtest[, 1]), ncol = h)
      for (i in 1:h) {
        if (i > 1) {
          xtest <- cbind(xtest[2:ncol(xtest)], y_hat_h)
        }
        colnames(xtest) <- private$train_colnames
        y_hat_h <- predict(private$fitted_model, as.matrix(xtest))
        pf[, i] <- y_hat_h
      }
      forecast <- NULL
      forecast$pf <- pf
      forecast
    }
  )
)

# DEEPNET AR MODEL-------------------------------------------------------------
DeepNetAR <- R6Class(
  classname = "DeepNetAR",
  private = list(fitted_model = NULL,
                 train_colnames = NULL),
  public = list(
    initialize = function(xtrain, ytrain) {
      dimx <- ncol(xtrain)
      deep_model <- keras_model_sequential() %>%
        layer_dense(
          units = 32,
          activation = 'relu',
          input_shape = c(dimx)
        ) %>%
        layer_dense(
          units = 32,
          activation = 'relu',
          input_shape = c(dimx)
        ) %>%
        layer_dense(
          units = 32,
          activation = 'relu',
          input_shape = c(dimx)
        ) %>%
        layer_dense(
          units = 32,
          activation = 'relu',
          input_shape = c(dimx)
        ) %>%
        layer_dense(
          units = 32,
          activation = 'relu',
          input_shape = c(dimx)
        ) %>%
        # layer_dense(units = 32, activation = 'relu', input_shape = c(ncol(W))) %>%
        # layer_dense(units = 32, activation = 'relu', input_shape = c(ncol(W))) %>%
        # layer_dense(units = 32, activation = 'relu', input_shape = c(ncol(W))) %>%
        # layer_dropout(0.15) %>%
        #    layer_batch_normalization() %>%
        layer_dense(units = 1, activation = 'linear') %>%
        compile(
          loss =  'mean_absolute_error',
          #abs_var_penal_loss, #
          optimizer =  optimizer_adam(),
          #optimizer_adam(lr = 0.001, decay = 0), #optimizer_sgd(lr = 0.00001*5, momentum = 0.0, nesterov = FALSE, decay = 0.05/nrow(xtrain)),
          metrics = c('mean_absolute_error')
        )
      sampind = sample(nrow(xtrain))
      xtrain = xtrain[sampind, , drop = FALSE]
      ytrain = ytrain[sampind]
      deep_model %>% fit(
        as.matrix(xtrain),
        as.matrix(ytrain),
        batch_size = 1024,
        epochs = 1000,
        validation_split = VAL_SIZE,
        #validation_data = list(xval, yval),
        view_metrics = FALSE,
        sample_weight = NULL,
        callbacks = list(
          callback_early_stopping(
            monitor = "val_mean_absolute_error",
            patience = PATIENCE,
            restore_best_weights = TRUE
          )
        )
      )
      deep_model$lag = dimx
      private$fitted_model <- deep_model
      private$train_colnames <- colnames(xtrain)
    },
    predict = function(xtest, h = 1) {
      pf  <- matrix(nrow = length(xtest[, 1]), ncol = h)
      for (i in 1:h) {
        if (i > 1) {
          xtest <- cbind(xtest[2:ncol(xtest)], y_hat_h)
        }
        colnames(xtest) <- private$train_colnames
        y_hat_h <- predict(private$fitted_model, as.matrix(xtest))
        pf[, i] <- y_hat_h
      }
      forecast <- NULL
      forecast$pf <- pf
      forecast
    }
  )
)


################################## LOCAL ######################################
# ARIMA -----------------------------------------------------------------------
local_methods.arima <- function(data, cores = 8) {
  res <- pblapply(data, function(s) {
    model <- forecast::auto.arima(s$pp$x)
    arima_f <- forecast(model, h = s$h)
    s$lforecast$auto_arima$level <- arima_f$level
    s$lforecast$auto_arima$upper <- arima_f$upper
    s$lforecast$auto_arima$mean  <- arima_f$mean
    s$lforecast$auto_arima$lower <- arima_f$lower
    s
  }, cl = cores)
  res
}

# ETS -------------------------------------------------------------------------
local_methods.ets <- function(data, cores = 8) {
  res <- pblapply(data, function(s) {
    model <- forecast::ets(s$pp$x)
    ets_f <- forecast(model, h = s$h)
    s$lforecast$ets$level <- ets_f$level
    s$lforecast$ets$upper <- ets_f$upper
    s$lforecast$ets$mean  <- ets_f$mean
    s$lforecast$ets$lower <- ets_f$lower
    s
  }, cl = cores)
  res
}
# THETA -----------------------------------------------------------------------
local_methods.theta <- function(data, cores = 8) {
  res <- pblapply(data, function(s) {
    theta_f <- forecast::thetaf(s$pp$x, h = s$h)
    s$lforecast$theta$level <- theta_f$level
    s$lforecast$theta$upper <- theta_f$upper
    s$lforecast$theta$mean <- theta_f$mean
    s$lforecast$theta$lower <- theta_f$lower
    s
  }, cl = cores)
  res
}
