# Snakefile — reproducible preprocessing for the PBMC cohort.
# Run from the repository root:
#
#   snakemake -c 1 -n      # dry run
#   snakemake -c 1         # execute (also creates logs/*.log via tee)
#
# Stages:
#   1. validate GEO .rds files exist under data/project_1/final_data
#   2. week1.R → results/{sample_sheet,week1_summary,seurat objects}
#   3. qc_summary.R → results/qc_summary.{csv,json}
#   4. export_for_geneformer.R → data/geneformer/raw/*

from pathlib import Path
import csv

configfile: "config/config.yaml"

DATA_DIR = Path(config["input_rds_dir"])
SAMPLES_TSV = Path(config["samples_tsv"])


def read_samples(tsv_path: Path) -> list:
    """Parse TSV rows. Lines starting with '#' are documentation (R does the same)."""
    data_lines = []
    with tsv_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data_lines.append(line.rstrip("\r\n"))

    if not data_lines:
        raise RuntimeError(f"No non-comment rows in {tsv_path}")

    reader = csv.DictReader(data_lines, delimiter="\t")
    fieldnames = reader.fieldnames
    if not fieldnames:
        raise RuntimeError(f"No header row in {tsv_path}")

    rows = []
    for raw in reader:
        row = {k: (raw.get(k, "") or "").strip() for k in fieldnames}
        sid = row.get("sample_id", "")
        if not sid or sid.startswith("#"):
            continue
        rows.append(row)
    return rows


SAMPLES = read_samples(SAMPLES_TSV)
if not SAMPLES:
    raise RuntimeError(f"No samples parsed from {SAMPLES_TSV}")

SAMPLE_IDS = [r["sample_id"] for r in SAMPLES]

RDS_INPUTS = [str(DATA_DIR / r["file_basename"]) for r in SAMPLES]


rule all:
    input:
        "results/week1_summary.csv",
        "results/sample_sheet.csv",
        "results/seurat_list_raw.rds",
        "results/sample_sheet.rds",
        "results/qc_summary.csv",
        "results/qc_summary.json",
        "data/geneformer/raw/obs.csv",
        "data/geneformer/raw/sample_sheet.csv",
        expand("data/geneformer/raw/{sid}/matrix.mtx", sid=SAMPLE_IDS),


rule validate_inputs:
    """Fail fast if GEO .rds files are absent from canonical directory."""
    input:
        samples_tsv=SAMPLES_TSV,
        rds=RDS_INPUTS,
    output:
        touch("results/.inputs_validated"),
    run:
        for r in SAMPLES:
            path = DATA_DIR / r["file_basename"]
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Missing {path} — populate {DATA_DIR} "
                    f"from GEO GSE150728 (see config/samples.tsv)."
                )


rule week1:
    input:
        samples_tsv=SAMPLES_TSV,
        flag="results/.inputs_validated",
        rds=RDS_INPUTS,
    output:
        "results/week1_summary.csv",
        "results/sample_sheet.csv",
        "results/seurat_list_raw.rds",
        "results/sample_sheet.rds",
    log:
        "logs/week1.log",
    shell:
        "mkdir -p logs results && "
        "Rscript week1.R 2>&1 | tee {log}"


rule qc_summary:
    input:
        "results/seurat_list_raw.rds",
        "results/sample_sheet.rds",
    output:
        "results/qc_summary.csv",
        "results/qc_summary.json",
    log:
        "logs/qc_summary.log",
    shell:
        "Rscript qc_summary.R 2>&1 | tee {log}"


rule export_geneformer:
    input:
        samples_tsv=SAMPLES_TSV,
        flag="results/.inputs_validated",
        rds=RDS_INPUTS,
    output:
        obs="data/geneformer/raw/obs.csv",
        sample_sheet="data/geneformer/raw/sample_sheet.csv",
        mtx=expand("data/geneformer/raw/{sid}/matrix.mtx", sid=SAMPLE_IDS),
    log:
        "logs/export_for_geneformer.log",
    shell:
        "mkdir -p logs data/geneformer/raw && "
        "Rscript export_for_geneformer.R 2>&1 | tee {log}"
