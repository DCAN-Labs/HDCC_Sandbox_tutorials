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

install.packages("arrow")      # Loading parquet files into memory
install.packages("corrplot")
install.packages("cifti")      # Loading CIFTI files (the CIFTI format is our standard for HBCD imaging output files)
install.packages("ciftiTools") # Tools for working with and visualizing CIFTI neuroimaging data
install.packages("data.table") # Helpful for fast data manipulation and reshaping large data frames
install.packages("dplyr")      # Helpful for looping through the same operations multiple times
install.packages("ggplot2")    # For data visualizations
install.packages("ggpubr")
install.packages("ggraph")
install.packages("ggsci")      # Provides scientific journal color schemes for ggplot2
install.packages("kableExtra")
install.packages("igraph")
install.packages("lme4")       # Linear mixed-effects models
install.packages("lmerTest")   # Adds p-values and tests to lme4 mixed models to predict outcomes
install.packages("matrixStats")
install.packages("naniar")
install.packages("nlme")       # Nonlinear mixed-effects models (alternative to lme4)
install.packages("plotrix")    # Additional plotting functions to generate heat maps
install.packages("pls")        # Partial least squares regression and PCA-related methods
install.packages("pwrss")
install.packages("reshape2")   # Reshaping data between vectors and matrices
install.packages("RSpectra")   # Efficient eigenvalue decomposition (data dimensionality reduction)
install.packages("tidyr")      # Tidying and reshaping data, looping multiple operations
install.packages("units")
install.packages("systemfonts")# Fonts for plots and graphics
install.packages("ragg")       # High-quality graphics rendering for ggplot
install.packages("pkgdown")    # Build static documentation websites from R packages
install.packages("textshaping")# Text rendering support for graphics (used with ggplot, ragg)
install.packages("nloptr")     # Nonlinear optimization algorithms
install.packages("openssl")    # Secure data transfer, encryption, and HTTPS connections
install.packages("devtools")   # Tools for developing, installing, and managing R packages
install.packages("forcats")    # Tools for working with categorical variables (factors)
install.packages("ggtext")     # Improved text formatting in ggplot (markdown, rich text)
install.packages("lubridate")  # Easy handling of dates and times
install.packages("magrittr")   # Provides the pipe operator (%>%) for cleaner code
install.packages("purrr")      # Functional programming tools for iteration and lists
install.packages("readr")      # Fast reading of CSV and text data files
install.packages("stringr")    # String/text processing
install.packages("tibble")     # Modern version of data frames
install.packages("sf")
install.packages("vdiffr")
install.packages("ggseg")
install.packages("ggseg3d")
install.packages("ggseg.meshes")
install.packages("ggseg.formats")
library(devtools)
install_github("PennLINC/ModelArray")
install_github("ggsegverse/ggsegFreeSurfer")
install_github("Jfortin1/neuroCombat_Rpackage")
