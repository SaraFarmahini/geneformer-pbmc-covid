options(repos = c(CRAN = "https://cloud.r-project.org"))

cat("== Pin spatstat satellite packages to versions compatible with spatstat.core 2.4-4 ==\n")

cat("Removing newer spatstat satellite packages...\n")
to_remove <- c("spatstat.utils", "spatstat.data", "spatstat.sparse",
               "spatstat.geom", "spatstat.random", "spatstat.explore",
               "spatstat.univar")
for (p in to_remove) {
  if (p %in% rownames(installed.packages()))
    try(remove.packages(p), silent = TRUE)
}

if (!requireNamespace("remotes", quietly = TRUE))
  install.packages("remotes")

cat("\n== Install old spatstat satellites from CRAN archive ==\n")
remotes::install_version("spatstat.utils",  version = "2.3-1",
                         repos = "https://cloud.r-project.org",
                         upgrade = "never")
remotes::install_version("spatstat.data",   version = "2.2-0",
                         repos = "https://cloud.r-project.org",
                         upgrade = "never")
remotes::install_version("spatstat.sparse", version = "2.1-1",
                         repos = "https://cloud.r-project.org",
                         upgrade = "never")
remotes::install_version("spatstat.geom",   version = "2.4-0",
                         repos = "https://cloud.r-project.org",
                         upgrade = "never")
remotes::install_version("spatstat.random", version = "2.2-0",
                         repos = "https://cloud.r-project.org",
                         upgrade = "never")

cat("\n== Re-install DoubletFinder ==\n")
remotes::install_github("chris-mcginnis-ucsf/DoubletFinder",
                        upgrade = "never")

cat("\n== Test loading DoubletFinder ==\n")
print(tryCatch({
  suppressPackageStartupMessages(library(DoubletFinder))
  "DoubletFinder loaded OK"
}, error = function(e) paste("FAIL:", e$message)))

cat("\n== Install SeuratWrappers compatible with Seurat 4.0.1 ==\n")
remotes::install_github("satijalab/seurat-wrappers",
                        ref = "10b4eb6c50f5dca2c0e9ae0a51df2b2c3e6a3d1e",
                        upgrade = "never")

cat("\n== DONE ==\n")
