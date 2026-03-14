##required dependencies for the HDCC tutorial 
#Sys.setenv(PKG_CONFIG_PATH = "/usr/local/ssl/lib/pkgconfig:/usr/local/ssl/lib64/pkgconfig",
#           CFLAGS = "-I/usr/local/ssl/include",
#           LDFLAGS = "-L/usr/local/ssl/lib -L/usr/local/ssl/lib64")
#nloptrpackageurl <- "https://cran.r-project.org/src/contrib/Archive/nloptr/nloptr_1.2.1.tar.gz"
#install.packages(nloptrpackageurl, repos=NULL, type="source")
#systemfontspackageurl <- "https://cran.r-project.org/src/contrib/Archive/systemfonts/systemfonts_1.2.3.tar.gz" 
#install.packages(systemfontspackageurl, repos=NULL, type="source")
#textshapingpackageurl <- "https://cran.r-project.org/src/contrib/Archive/textshaping/textshaping_1.0.1.tar.gz"
#install.packages(textshapingpackageurl,repos=NULL,type="source")
#raggpackageurl <- "https://cran.r-project.org/src/contrib/Archive/ragg/ragg_1.3.2.tar.gz" 
#install.packages(raggpackageurl, repos=NULL, type="source")
#pkgdownpackageurl <- "https://cran.r-project.org/src/contrib/Archive/pkgdown/pkgdown_2.1.3.tar.gz" 
#install.packages(pkgdownpackageurl, repos=NULL, type="source")
#arrowpackageurl <- "https://cran.r-project.org/src/contrib/Archive/arrow/arrow_22.0.0.tar.gz"
#install.packages(arrowpackageurl,repos=NULL,type="source")
install.packages("arrow")
install.packages("systemfonts")
install.packages("ragg")
install.packages("pkgdown")
install.packages("textshaping")
install.packages("nloptr")
install.packages("openssl")
install.packages("cifti")
install.packages("data.table")
install.packages("devtools")
install.packages("dplyr")
install.packages("forcats")
install.packages("ggplot2")
install.packages("ggsci")
install.packages("ggtext")
install.packages("lme4")
install.packages("lmerTest")
install.packages("lubridate")
install.packages("magrittr")
install.packages("nlme")
install.packages("plotrix")
install.packages("pls")
install.packages("purrr")
install.packages("readr")
install.packages("reshape2")
install.packages("RSpectra")
install.packages("stringr")
install.packages("tidyr")
install.packages("tibble")
library(devtools)
install_github("PennLINC/ModelArray")
