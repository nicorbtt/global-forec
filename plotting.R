## ----------------------------------------------------------------------------
##
## Script name: plotting
## Purpose of script: forecasts and error visualization
##
## Author: Nicolo' Rubattu
##
## Date Created: 12-10-2022
## Copyright (c) Nicolo' Rubattu, 2022
## Email: nicolo.rubattu@idsia.ch
## ----------------------------------------------------------------------------


library(ggplot2)

# Given metrics tables, and models forecasts, plot them -----------------------
plotting.plot_metrics <- function(tables) {
  plots <- list()
  table_i <- 1
  for (table in tables) {
    tmp <- NULL
    tmp$metric <- table$name
    table_df <- as.data.frame(table$tab)
    colnames(table_df) <- c("h", "Model", "Metric")
    new_h = vector(length = length(table_df$h))
    for (i in 1:length(table_df$h)) {
      new_h[i] <-
        strsplit(x = as.character(table_df$h[[i]]), split = "=")[[1]][2]
    }
    table_df$h <- as.numeric(new_h)
    p <-
      ggplot(data = table_df, aes(x = h, y = Metric, group = Model)) +
      geom_line(aes(color = Model)) +
      geom_point(aes(color = Model)) +
      ylab(table$name)
    tmp$p <- p
    plots[[table_i]] <- tmp
    table_i = table_i + 1
  }
  plots
}
