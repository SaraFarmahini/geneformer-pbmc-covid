###############################################################################
## Quick inspector for the project's per-sample .rds files.
## Usage:
##   Rscript inspect_rds.R                     # briefly inspect all .rds files under data/project_1/final_data
##   Rscript inspect_rds.R <path/to/file.rds>  # inspects one file in detail
###############################################################################
suppressPackageStartupMessages(library(Matrix))

DEFAULT_DIR <- "data/project_1/final_data"

args <- commandArgs(trailingOnly = TRUE)
files <- if (length(args)) args else list.files(DEFAULT_DIR, "\\.rds$", full.names = TRUE)

human_int <- function(n) format(n, big.mark = ",", scientific = FALSE)

inspect_one <- function(file, detail = FALSE) {
  cat("\n##", basename(file), "\n")
  x <- readRDS(file)

  cat("  top-level class :", paste(class(x), collapse = "/"), "\n")
  if (is.list(x)) {
    cat("  list elements   :", paste(names(x), collapse = ", "), "\n")
    for (nm in names(x)) {
      m <- x[[nm]]
      d <- dim(m)
      nnz <- if (inherits(m, "dgCMatrix")) length(m@x) else sum(m != 0)
      density <- 100 * nnz / prod(as.numeric(d))
      cat(sprintf("    $%-9s : %s [%s genes x %s cells, %s nnz, %.2f%% dense]\n",
                  nm, paste(class(m), collapse = "/"),
                  human_int(d[1]), human_int(d[2]),
                  human_int(nnz), density))
    }

    if (detail) {
      m <- x[[1]]
      cat("\n  --- detail on $", names(x)[1], " ---\n", sep = "")
      cat("  first 5 genes : ",
          paste(head(rownames(m), 5), collapse = ", "), "\n")
      cat("  first 5 cells: ",
          paste(head(colnames(m), 5), collapse = ", "), "\n\n")
      cat("  5 x 5 corner:\n")
      print(as.matrix(m[1:5, 1:5]))

      counts_per_cell <- Matrix::colSums(m)
      genes_per_cell  <- Matrix::colSums(m > 0)
      cat("\n  per-cell UMI total : ",
          sprintf("median=%s, IQR=[%s, %s], max=%s",
                  human_int(median(counts_per_cell)),
                  human_int(quantile(counts_per_cell, .25)),
                  human_int(quantile(counts_per_cell, .75)),
                  human_int(max(counts_per_cell))), "\n")
      cat("  per-cell genes seen: ",
          sprintf("median=%s, IQR=[%s, %s], max=%s",
                  human_int(median(genes_per_cell)),
                  human_int(quantile(genes_per_cell, .25)),
                  human_int(quantile(genes_per_cell, .75)),
                  human_int(max(genes_per_cell))), "\n")
    }
  } else {
    cat("  (not a list — printing str())\n")
    str(x, max.level = 1)
  }
}

if (length(args) == 1) {
  inspect_one(args[[1]], detail = TRUE)
} else {
  for (f in files) inspect_one(f, detail = FALSE)
}
