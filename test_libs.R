cat("Testing library loads...\n\n")

libs <- c("dplyr", "spatstat.core", "Seurat", "patchwork",
          "DoubletFinder", "SingleR", "enrichR",
          "SingleCellExperiment", "SeuratWrappers", "tidyverse",
          "celldex")

for (lib in libs) {
  result <- tryCatch({
    suppressPackageStartupMessages(library(lib, character.only = TRUE))
    "OK"
  }, error = function(e) paste("FAIL:", e$message))
  cat(sprintf("%-22s : %s\n", lib, result))
}
