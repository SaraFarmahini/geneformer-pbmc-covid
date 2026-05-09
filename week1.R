###############################################################################
## Single Cell Bioinformatics — Project 1, Week 1
##
## Loads every sample listed in config/samples.tsv, builds Seurat objects,
## writes unified sample_sheet + summaries.
##
## Dataset: Wilk et al. 2020, GSE150728 (PBMCs). Each .rds is a dropEst triple
## ($exon, $intron, $spanning); we standardise on $exon (mRNA).
###############################################################################

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(dplyr)
})

CONFIG_SAMPLES <- Sys.getenv("PROJECT_SAMPLES_TSV", "config/samples.tsv")
DATA_DIR <- "data/project_1/final_data"
OUT_DIR  <- "results"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

if (!file.exists(CONFIG_SAMPLES)) {
  stop(sprintf("Missing %s — add config/samples.tsv to the repo clone.", CONFIG_SAMPLES))
}

###############################################################################
## Sample sheet driven by TSV — single reproducible cohort definition
###############################################################################
sample_sheet <- read.delim(
  CONFIG_SAMPLES,
  sep               = "\t",
  stringsAsFactors  = FALSE,
  comment.char      = "#",
  quote             = ""
)

needed <- c("sample_id", "file_basename", "geo_id", "manuscript_id",
            "condition", "tissue", "study")
missing_cols <- setdiff(needed, names(sample_sheet))
if (length(missing_cols))
  stop("samples table missing columns: ", paste(missing_cols, collapse = ", "))

cat("\n=== Sample sheet (from ", CONFIG_SAMPLES, ") ===\n", sep = "")
print(sample_sheet, row.names = FALSE)

write.csv(sample_sheet, file.path(OUT_DIR, "sample_sheet.csv"), row.names = FALSE)

###############################################################################
## Load each RDS, extract exon counts, build Seurat, prefix barcodes globally
###############################################################################
seurat_list <- vector("list", nrow(sample_sheet))
names(seurat_list) <- sample_sheet$sample_id

for (i in seq_len(nrow(sample_sheet))) {
  sid  <- sample_sheet$sample_id[i]
  file <- file.path(DATA_DIR, sample_sheet$file_basename[i])
  if (!file.exists(file))
    stop("Missing RDS for ", sid, ": ", file, "\n",
         "Populate data/project_1/final_data/ with GEO artefacts listed in ",
         CONFIG_SAMPLES)

  cat(sprintf("\n[%d/%d] Loading %s ...\n", i, nrow(sample_sheet), basename(file)))

  raw  <- readRDS(file)
  cnts <- raw$exon
  cat(sprintf("  raw matrix: %d genes x %d cells\n", nrow(cnts), ncol(cnts)))

  colnames(cnts) <- paste(sid, colnames(cnts), sep = "_")

  obj <- CreateSeuratObject(
    counts       = cnts,
    project      = sid,
    min.cells    = 0,
    min.features = 0
  )

  for (col in setdiff(names(sample_sheet), "file_basename"))
    obj[[col]] <- sample_sheet[i, col]

  seurat_list[[sid]] <- obj
}

###############################################################################
## Cells / genes / metadata per sample
###############################################################################
report <- do.call(rbind, lapply(names(seurat_list), function(sid) {
  obj <- seurat_list[[sid]]
  data.frame(
    sample_id  = sid,
    cells      = ncol(obj),
    genes      = nrow(obj),
    meta_cols  = paste(colnames(obj@meta.data), collapse = ", "),
    stringsAsFactors = FALSE
  )
}))

cat("\n=== Cells / genes / metadata per sample ===\n")
print(report, row.names = FALSE)

write.csv(report, file.path(OUT_DIR, "week1_summary.csv"), row.names = FALSE)

saveRDS(seurat_list,   file.path(OUT_DIR, "seurat_list_raw.rds"))
saveRDS(sample_sheet,  file.path(OUT_DIR, "sample_sheet.rds"))

cat(sprintf(
  "\nSaved %d Seurat objects to %s/seurat_list_raw.rds\n",
  length(seurat_list), OUT_DIR
))
cat("Week 1 complete.\n")
