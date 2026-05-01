###############################################################################
## Export the 4 PBMC samples from .rds (Wilk et al. 2020, GSE150728)
## into Matrix-Market format suitable for loading into AnnData/Geneformer.
##
## For each sample we write:
##   data/geneformer/raw/<sample_id>/matrix.mtx        (genes x cells, sparse)
##   data/geneformer/raw/<sample_id>/features.tsv      (gene symbols)
##   data/geneformer/raw/<sample_id>/barcodes.tsv      (cell barcodes, sample-prefixed)
##
## Plus combined metadata:
##   data/geneformer/raw/obs.csv          (one row per cell)
##   data/geneformer/raw/sample_sheet.csv (one row per sample)
###############################################################################
suppressPackageStartupMessages(library(Matrix))

DATA_DIR <- "data/project_1/final_data"
OUT_ROOT <- "data/geneformer/raw"
dir.create(OUT_ROOT, recursive = TRUE, showWarnings = FALSE)

samples <- list(
  S556   = list(rds = "GSM4557329_GSM4557329_556_cell.counts.matrices.rds",
                manuscript = "C2",  condition = "COVID-19"),
  S557   = list(rds = "GSM4557330_GSM4557330_557_cell.counts.matrices.rds",
                manuscript = "C3",  condition = "COVID-19"),
  S558   = list(rds = "GSM4557331_GSM4557331_558_cell.counts.matrices.rds",
                manuscript = "C4",  condition = "COVID-19"),
  HIP043 = list(rds = "GSM4557337_GSM4557337_HIP043_cell.counts.matrices.rds",
                manuscript = "H4",  condition = "Healthy")
)

obs_list <- list()
total_cells <- 0L

for (sid in names(samples)) {
  cfg <- samples[[sid]]
  rds_path <- file.path(DATA_DIR, cfg$rds)
  out_dir  <- file.path(OUT_ROOT, sid)
  dir.create(out_dir, showWarnings = FALSE)

  cat(sprintf("[%s] reading %s\n", sid, basename(rds_path)))
  x <- readRDS(rds_path)
  m <- x$exon  # genes x cells

  # prefix barcodes so they are globally unique
  prefixed_barcodes <- paste0(sid, "_", colnames(m))
  colnames(m) <- prefixed_barcodes

  cat(sprintf("[%s] writing %s genes x %s cells\n",
              sid,
              format(nrow(m), big.mark = ","),
              format(ncol(m), big.mark = ",")))
  writeMM(m, file = file.path(out_dir, "matrix.mtx"))
  writeLines(rownames(m),    con = file.path(out_dir, "features.tsv"))
  writeLines(colnames(m),    con = file.path(out_dir, "barcodes.tsv"))

  obs_list[[sid]] <- data.frame(
    barcode       = colnames(m),
    sample_id     = sid,
    manuscript_id = cfg$manuscript,
    condition     = cfg$condition,
    n_counts      = as.integer(Matrix::colSums(m)),
    n_genes       = as.integer(Matrix::colSums(m > 0)),
    stringsAsFactors = FALSE
  )
  total_cells <- total_cells + ncol(m)
}

obs_all <- do.call(rbind, obs_list)
rownames(obs_all) <- NULL
write.csv(obs_all,
          file.path(OUT_ROOT, "obs.csv"),
          row.names = FALSE,
          quote     = FALSE)

sample_df <- data.frame(
  sample_id     = names(samples),
  manuscript_id = sapply(samples, `[[`, "manuscript"),
  condition     = sapply(samples, `[[`, "condition"),
  n_cells       = sapply(obs_list, nrow),
  stringsAsFactors = FALSE
)
write.csv(sample_df,
          file.path(OUT_ROOT, "sample_sheet.csv"),
          row.names = FALSE,
          quote     = FALSE)

cat(sprintf("\nDONE. %s cells across %d samples written to %s\n",
            format(total_cells, big.mark = ","),
            length(samples),
            OUT_ROOT))
print(sample_df)
