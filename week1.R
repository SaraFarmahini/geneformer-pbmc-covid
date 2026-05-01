###############################################################################
## Single Cell Bioinformatics 2025-26 -- Project 1, Week 1
## Tasks: 1) Load data & build Seurat objects
##        2) Sample sheet (metadata from Table 1)
##        3) Report cells / genes / metadata per sample
##
## Dataset: Wilk et al. 2020, GSE150728 (PBMCs from COVID-19 patients & healthy
## controls). Each .rds file is a list of three sparse count matrices produced
## by dropEst: $exon, $intron, $spanning. We use $exon for the Seurat object
## (standard mRNA counts; the intron/spanning matrices are for RNA-velocity).
###############################################################################

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(dplyr)
})

DATA_DIR <- "data/project_1/final_data"
OUT_DIR  <- "results"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

###############################################################################
## 2) Sample sheet -- metadata from Table 1 (Wilk et al. 2020 / GSE150728).
##    The four samples in the project are PBMCs from 3 COVID-19 patients and
##    1 healthy control. Manuscript IDs (C2, C3, C4, H4) follow Wilk et al.
##    Update the columns below with anything additional that Table 1 lists
##    (age, sex, severity, ...).
###############################################################################

sample_sheet <- data.frame(
  sample_id      = c("S556", "S557", "S558", "HIP043"),
  file_basename  = c("GSM4557329_GSM4557329_556_cell.counts.matrices.rds",
                     "GSM4557330_GSM4557330_557_cell.counts.matrices.rds",
                     "GSM4557331_GSM4557331_558_cell.counts.matrices.rds",
                     "GSM4557337_GSM4557337_HIP043_cell.counts.matrices.rds"),
  geo_id         = c("GSM4557329", "GSM4557330", "GSM4557331", "GSM4557337"),
  manuscript_id  = c("C2", "C3", "C4", "H4"),
  condition      = c("COVID-19", "COVID-19", "COVID-19", "Healthy"),
  tissue         = "PBMC",
  study          = "Wilk et al. 2020 (GSE150728)",
  stringsAsFactors = FALSE
)

cat("\n=== Sample sheet ===\n")
print(sample_sheet, row.names = FALSE)

write.csv(sample_sheet, file.path(OUT_DIR, "sample_sheet.csv"), row.names = FALSE)

###############################################################################
## 1) Load each RDS, extract the exon count matrix, build a Seurat object.
##    Cell barcodes within a sample are short and may not be unique across
##    samples, so we prepend the sample_id during object creation. Seurat's
##    CreateSeuratObject argument `project` becomes the `orig.ident` of every
##    cell -- exactly what we want as a per-sample label.
###############################################################################

seurat_list <- vector("list", nrow(sample_sheet))
names(seurat_list) <- sample_sheet$sample_id

for (i in seq_len(nrow(sample_sheet))) {
  sid  <- sample_sheet$sample_id[i]
  file <- file.path(DATA_DIR, sample_sheet$file_basename[i])
  cat(sprintf("\n[%d/%d] Loading %s ...\n", i, nrow(sample_sheet), basename(file)))

  raw  <- readRDS(file)
  cnts <- raw$exon
  cat(sprintf("  raw matrix: %d genes x %d cells\n", nrow(cnts), ncol(cnts)))

  ## Make cell barcodes globally unique by prefixing with sample id
  colnames(cnts) <- paste(sid, colnames(cnts), sep = "_")

  obj <- CreateSeuratObject(
    counts       = cnts,
    project      = sid,
    min.cells    = 0,   # do not filter genes here -- filtering is a Week-2 task
    min.features = 0    # idem for cells
  )

  ## 3) Add metadata: copy every column from the sample sheet onto each cell
  for (col in setdiff(colnames(sample_sheet), "file_basename")) {
    obj[[col]] <- sample_sheet[i, col]
  }

  seurat_list[[sid]] <- obj
}

###############################################################################
## 3) Report number of cells, number of genes, metadata columns.
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

###############################################################################
## Save Seurat objects for re-use in later weeks
###############################################################################

saveRDS(seurat_list,  file.path(OUT_DIR, "seurat_list_raw.rds"))
saveRDS(sample_sheet, file.path(OUT_DIR, "sample_sheet.rds"))

cat(sprintf(
  "\nSaved %d Seurat objects to %s/seurat_list_raw.rds\n",
  length(seurat_list), OUT_DIR
))
cat("Week 1 complete.\n")
