###############################################################################
## Export every sample from config/samples.tsv into MatrixMarket + Geneformer-ready obs.
###############################################################################
suppressPackageStartupMessages({
  library(Matrix)
})

CONFIG_SAMPLES <- Sys.getenv("PROJECT_SAMPLES_TSV", "config/samples.tsv")
DATA_DIR <- "data/project_1/final_data"
OUT_ROOT <- "data/geneformer/raw"
dir.create(OUT_ROOT, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(CONFIG_SAMPLES)) {
  stop(sprintf("Missing %s", CONFIG_SAMPLES))
}

samples_tbl <- read.delim(
  CONFIG_SAMPLES,
  sep               = "\t",
  stringsAsFactors  = FALSE,
  comment.char      = "#",
  quote             = ""
)

required <- c(
  "sample_id", "file_basename",
  "geo_id", "manuscript_id", "condition", "tissue", "study"
)
miss <- setdiff(required, names(samples_tbl))
if (length(miss))
  stop("samples table missing columns: ", paste(miss, collapse = ", "))

samples <- setNames(vector("list", nrow(samples_tbl)), samples_tbl$sample_id)
for (i in seq_len(nrow(samples_tbl))) {
  sid <- samples_tbl$sample_id[i]
  samples[[sid]] <- list(
    rds         = samples_tbl$file_basename[i],
    manuscript  = samples_tbl$manuscript_id[i],
    condition   = samples_tbl$condition[i],
    geo_id      = samples_tbl$geo_id[i],
    tissue      = samples_tbl$tissue[i],
    study       = samples_tbl$study[i]
  )
}

obs_list <- list()
total_cells <- 0L

for (sid in names(samples)) {
  cfg <- samples[[sid]]
  rds_path <- file.path(DATA_DIR, cfg$rds)
  if (!file.exists(rds_path))
    stop(sprintf("RDS missing for %s: %s", sid, rds_path))
  out_dir  <- file.path(OUT_ROOT, sid)
  dir.create(out_dir, showWarnings = FALSE)

  cat(sprintf("[%s] reading %s\n", sid, basename(rds_path)))
  x <- readRDS(rds_path)
  m <- x$exon

  prefixed_barcodes <- paste0(sid, "_", colnames(m))
  colnames(m) <- prefixed_barcodes

  cat(sprintf("[%s] writing %s genes x %s cells\n",
              sid,
              format(nrow(m), big.mark = ","),
              format(ncol(m), big.mark = ",")))
  Matrix::writeMM(m, file = file.path(out_dir, "matrix.mtx"))
  writeLines(rownames(m), con = file.path(out_dir, "features.tsv"))
  writeLines(colnames(m), con = file.path(out_dir, "barcodes.tsv"))

  obs_list[[sid]] <- data.frame(
    barcode       = colnames(m),
    sample_id     = sid,
    manuscript_id = cfg$manuscript,
    condition     = cfg$condition,
    geo_id        = cfg$geo_id,
    tissue        = cfg$tissue,
    study         = cfg$study,
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
          quote = FALSE)

sample_df <- samples_tbl[, c(
  "sample_id", "geo_id", "manuscript_id", "condition", "tissue", "study")]
sample_df$n_cells <- sapply(samples_tbl$sample_id, function(sid) nrow(obs_list[[sid]]))

write.csv(sample_df,
          file.path(OUT_ROOT, "sample_sheet.csv"),
          row.names = FALSE,
          quote = FALSE)

cat(sprintf("\nDONE. %s cells across %d samples written to %s\n",
            format(total_cells, big.mark = ","),
            length(samples),
            OUT_ROOT))
print(sample_df)
