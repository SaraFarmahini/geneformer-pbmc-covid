###############################################################################
## QC summary across the 4 samples -- basic per-cell metrics for visualization
###############################################################################
suppressPackageStartupMessages({
  library(Matrix); library(Seurat); library(dplyr); library(jsonlite)
})

seurat_list <- readRDS("results/seurat_list_raw.rds")
sample_sheet <- readRDS("results/sample_sheet.rds")

mt_genes_pattern <- "^MT-"

per_sample <- lapply(names(seurat_list), function(sid) {
  obj <- seurat_list[[sid]]
  obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = mt_genes_pattern)
  meta <- sample_sheet[match(sid, sample_sheet$sample_id), ]
  list(
    sample_id      = sid,
    geo_id         = meta$geo_id,
    manuscript_id  = meta$manuscript_id,
    condition      = meta$condition,
    cells          = ncol(obj),
    genes          = nrow(obj),
    median_umi     = round(median(obj$nCount_RNA)),
    median_genes   = round(median(obj$nFeature_RNA)),
    median_mt_pct  = round(median(obj[["percent.mt"]][, 1]), 2),
    q25_umi        = round(as.numeric(quantile(obj$nCount_RNA, 0.25))),
    q75_umi        = round(as.numeric(quantile(obj$nCount_RNA, 0.75))),
    q25_genes      = round(as.numeric(quantile(obj$nFeature_RNA, 0.25))),
    q75_genes      = round(as.numeric(quantile(obj$nFeature_RNA, 0.75)))
  )
})

names(per_sample) <- names(seurat_list)

cat("\n=== Per-sample QC summary ===\n")
df <- do.call(rbind, lapply(per_sample, function(x) {
  data.frame(
    sample_id     = x$sample_id,
    manuscript_id = x$manuscript_id,
    condition     = x$condition,
    cells         = x$cells,
    genes         = x$genes,
    median_umi    = x$median_umi,
    median_genes  = x$median_genes,
    median_mt_pct = x$median_mt_pct,
    stringsAsFactors = FALSE
  )
}))
print(df, row.names = FALSE)

write.csv(df, "results/qc_summary.csv", row.names = FALSE)
write_json(per_sample, "results/qc_summary.json", pretty = TRUE, auto_unbox = TRUE)

cat("\nWrote results/qc_summary.csv and results/qc_summary.json\n")
