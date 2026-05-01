options(repos = c(CRAN = "https://cloud.r-project.org"))

cat("== R version & platform ==\n")
print(R.version.string)
print(Sys.info()[c("sysname", "machine")])
cat("Compilers visible to R:\n")
print(Sys.which(c("clang", "clang++", "gfortran",
                  "x86_64-apple-darwin13.4.0-clang",
                  "x86_64-apple-darwin13.4.0-gfortran")))

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if (!requireNamespace("remotes", quietly = TRUE))
  install.packages("remotes")

cat("\n== Installing SingleR (Bioconductor) ==\n")
BiocManager::install("SingleR", update = FALSE, ask = FALSE)

cat("\n== Installing DoubletFinder (GitHub) ==\n")
remotes::install_github("chris-mcginnis-ucsf/DoubletFinder", upgrade = "never")

cat("\n== Installing SeuratWrappers (GitHub, Seurat-4 compatible) ==\n")
if (!requireNamespace("R.utils", quietly = TRUE))
  install.packages("R.utils")
remotes::install_github("satijalab/seurat-wrappers",
                        ref = "d28512f804d5fe05e6d68900ca9221020d52cf1d",
                        upgrade = "never")

cat("\n== DONE ==\n")
