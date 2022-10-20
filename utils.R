## ----------------------------------------------------------------------------
##
## Script name: utils
## Purpose of script: utility functions for time series normalization and
##                    dataset loading and preparation
##
## Author: Nicolo' Rubattu
##
## Date Created: 12-10-2022
## Copyright (c) Nicolo' Rubattu, 2022
## Email: nicolo.rubattu@idsia.ch
## ----------------------------------------------------------------------------



source("models.R")

# Time series normalization: y = (x - min) / (max - min) ----------------------
utils.normalize <- function(serie, params = NULL) {
  if (is.null(params)) {
    min_value <- min(serie)
    max_value <- max(serie)
  } else {
    min_value <- params$min
    max_value <- params$max
  }
  res <- NULL
  res$x <- (serie - min_value) / (max_value - min_value)
  res$min <- min_value
  res$max <- max_value
  return(res)
}

utils.denormalize <- function(serie, params) {
  return(serie * (params$max - params$min) + params$min)
}

# Time series MASE scaling ----------------------------------------------------
utils.mase_normalize <- function(serie, params = NULL) {
  if (is.null(params)) {
    frq = floor(stats::frequency(serie))
    if (length(serie) < frq) {
      frq = 1
    }
    mase_scal <- mean(abs(
      utils::head(as.vector(serie),-frq) - utils::tail(as.vector(serie),-frq)
    ) ** 1)
  } else {
    mase_scal <- params$mase_scal
  }
  res <- NULL
  res$x <- serie / mase_scal
  res$mase_scal <- mase_scal
  return(res)
}

utils.mase_denormalize <- function(serie, params) {
  return(serie * params$mase_scal)
}

# Time series standardization: y = (x - mean) / sd ----------------------------
utils.standardize <- function(serie, params = NULL) {
  if (is.null(params)) {
    serie_mean <- mean(serie)
    serie_sd <- sd(serie)
  } else {
    serie_mean <- params$mean
    serie_sd <- params$sd
  }
  res <- NULL
  res$x <- (serie - serie_mean) / serie_sd
  res$mean <- serie_mean
  res$sd <- serie_sd
  return(res)
}

# DATASET MANAGEMENT ----------------------------------------------------------
DATASET_FOLDER = "./data/"
DATASET_LIST = c(
  "M1",
  "M3",
  "Tourism",
  "M4",
  "NN5",
  "Wikipedia",
  "FRED-MD",
  "Weather",
  "Dominick",
  "Traffic",
  "Electricity",
  "CarParts",
  "Hospital",
  "CIF2016",
  "Pedestrian",
  "DoublePendulum",
  "COVID19"
)

# Load dataset function
utils.load_dataset <- function(dname) {
  if (dname %in% DATASET_LIST) {
    print(paste("Loading dataset", dname, "..."))
    return(readRDS(paste0(DATASET_FOLDER, dname, ".rds")))
  } else {
    stop("Dataset does not exist!")
  }
}

# Prepare dataset function
utils.prepare_dataset <- function(dname, norm_mode = "mase_normalize") {
    # 1) Load dataset and check format ------------------------------------------
    data <- utils.load_dataset(dname)
    print(paste("#",length(data), " time series"))
    for (i in 1:length(data)) {
      data[[i]]$ID <- i
    }
    for (a in c("ID", "h", "x", "xx")) {
      if (!a %in% attributes(data[[1]])$names) {
        stop(paste0("attribute $", a, " doesn't exist!"))
      }
    }
    # 2) Normalization ----------------------------------------------------------
    for (i in 1:length(data)) {
      serie <- data[[i]]
      if (norm_mode == "normalize") {
        serie$pp <- utils.normalize(serie$x)
        serie$pp$xx <- utils.normalize(serie$xx, serie$pp)$x
      } else if (norm_mode == "standardize") {
        serie$pp <- utils.standardize(serie$x)
        serie$pp$xx <-
          utils.standardize(serie$xx, serie$pp)$x
      } else if (norm_mode == "mase_normalize") {
        serie$pp <- utils.mase_normalize(serie$x)
        serie$pp$xx <-
          utils.mase_normalize(serie$xx, serie$pp)$x
      } else if (is.null(norm_mode)) {
        serie$pp$x <- serie$x
        serie$pp$xx <- serie$xx
      } else {
        stop(
          "wrong normalization mode. chose one in {normalize, mase_normalize, standardize, NULL}"
        )
      }
      data[[i]] <- serie
    }
    # 3) Local Methods {AUTO.ARIMA, ETS, THETA} ---------------------------------
    print("Computing local methods (loops) - AUTO.ARIMA, ETS, THETA")
    data <- local_methods.arima(data, cores = 8)
    data <- local_methods.ets(data, cores = 8)
    data <- local_methods.theta(data, cores = 8)
    # 4) Save dataset -----------------------------------------------------------
    saveRDS(data, file = paste0(DATASET_FOLDER, dname, ".rds"))
    print(paste0("Dataset saved at ", DATASET_FOLDER, dname, ".rds"))
}

#utils.prepare_dataset("<dname>")

# Build lagged dataset of lag LAG
utils.build_lagged_dataset <- function(data, LAG, h, verbose = TRUE) {
  # N.B.
  # - only keep time series which length is greater than LAG
  # - h should be the same for each time series, i.e length(xx)
  data <-
    data[as.vector(unlist(lapply(data, function(s)
      length(s$x) > LAG)))]
  if (verbose)
    print(paste0("Processed #", length(data), " time series"))
  lagged_list <- list()
  for (i in 1:length(data)) {
    lagged_list[[i]] <- embed(data[[i]]$pp$x, (LAG + 1))[, (LAG + 1):1]
  }
  XY <- do.call(rbind, lagged_list)
  colnames(XY) <- c(paste("X", 1:LAG, sep = ""), 'y')
  X_train  <- XY[, 1:LAG]
  y_train  <- XY[, (LAG + 1)]
  X_test <- matrix(nrow = length(data), ncol = LAG)
  Y_test <- matrix(nrow = length(data), ncol = h)
  for (i in 1:length(data)) {
    X_test[i, ] <- as.numeric(tail(data[[i]]$pp$x, LAG))
    Y_test[i, ] <- as.numeric(data[[i]]$pp$xx)
  }
  if (is.numeric(X_train)) {
    X_train <- as.matrix(X_train, nrow=length(X_train), ncol=1)
  }
  colnames(X_test) <- colnames(XY)[1:LAG]
  rm(lagged_list, XY)
  laggedds <- NULL
  laggedds$data <- data
  laggedds$X_train <- X_train
  laggedds$y_train <- y_train
  laggedds$X_test <- X_test
  laggedds$Y_test <- Y_test
  laggedds
}
