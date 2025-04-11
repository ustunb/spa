# Reporting Helper Functions

required_packages = c('dplyr', 'ggplot2', 'tidyverse')
for (pkg in required_packages){
    suppressPackageStartupMessages(library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE));
}

sanitize_text <- function(x) {
    x = sanitize(x, type = "latex")
    x = gsub("_", "\\\\_", x)
    return(x)
}

sanitize_colnames <- function(x) {
    x = sanitize(x, type = "latex")
    x = gsub("_", "\\\\_", x)
    x = paste0("{\\bfseries ", x, "}")
    return(x)
}

human.numbers = function(x = NULL, smbl =""){
    #https://github.com/fdryan/R/blob/master/ggplot2_formatter.r
    humanity <- function(y){

        if (!is.na(y)){

            b <- round(abs(y) / 1e9, 0.1)
            m <- round(abs(y) / 1e6, 0.1)
            k <- round(abs(y) / 1e3, 0.1)

            if ( y >= 0 ){
                y_is_positive <- ""
            } else {
                y_is_positive <- "-"
            }

            if ( k < 1 ) {
                paste0(y_is_positive, smbl, y )
            } else if ( m < 1){
                paste0 (y_is_positive, smbl,  k , "K")
            } else if (b < 1){
                paste0 (y_is_positive, smbl, m ,"M")
            } else {
                paste0 (y_is_positive, smbl,  comma(b), "N")
            }
        }
    }
    sapply(x,humanity)
}

line_color = "#E9E9E9"
default_plot_theme = theme_bw() +
    theme(title = element_text(size = 18),
          plot.margin = margin(t = 0.25, r = 0, b = 0.75, l = 0.25, unit = "cm"),
          axis.line = element_blank(),
          panel.border = element_rect(linewidth = 2.0, color = line_color),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(linetype="solid", linewidth = 1.0, color=line_color),
          #
          axis.title.x = element_text(size = 20, margin = margin(t = 20, unit = "pt")),
          axis.text.x   = element_text(size = 20),
          axis.ticks.x  = element_line(linewidth = 1.0, color = line_color),
          #
          axis.title.y = element_text(size = 20, margin = margin(b = 20, unit = "pt")),
          axis.text.y   = element_text(size=20),
          axis.ticks.y	= element_line(linewidth = 1.0, color = line_color),
          axis.line.y = element_blank(),
          #
          #legend.position="none",
          legend.title = element_blank(),
          legend.text = element_text(face="plain",size=14,angle=0,lineheight=30),
          #legend.key.width = unit(1.5, "cm"),
          #legend.key.height = unit(1.5, "cm"),
          #legend.text.align = 0,
          legend.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "cm"))
