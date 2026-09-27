# Native comparators on the frozen full-transcriptome count matrices.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 1L)
root <- normalizePath(args[[1]], mustWork=TRUE)
suppressPackageStartupMessages({library(DESeq2); library(edgeR); library(jsonlite)})
cfg <- read_json(file.path(root, "protocol.json"), simplifyVector=TRUE)
stopifnot(as.character(packageVersion("DESeq2")) == cfg$native$DESeq2,
          as.character(packageVersion("edgeR")) == cfg$native$edgeR,
          as.character(getRversion()) == cfg$native$R)
write_tsv <- function(x, name) write.table(x, file.path(root, name), sep="\t",
                                           quote=FALSE, row.names=FALSE, na="NA")
counts <- as.matrix(read.delim(gzfile(file.path(root, "counts.tsv.gz")), row.names=1, check.names=FALSE))
meta <- read.delim(file.path(root, "samples.tsv"), check.names=FALSE)
features <- read.delim(file.path(root, "features.tsv"), check.names=FALSE)
stopifnot(identical(colnames(counts), meta$sample), identical(rownames(counts), features$gene),
          !anyDuplicated(features$gene), all(counts >= 0), all(counts == round(counts)),
          all(colSums(counts) > 0))
keep <- features$tested == 1L
meta$condition <- factor(meta$condition, levels=c("control", "case"))
rownames(meta) <- meta$sample
native_rle <- estimateSizeFactorsForMatrix(counts)
native_rle <- native_rle/exp(mean(log(native_rle)))
stopifnot(max(abs(log(native_rle/meta$rle_factor))) < 2e-12)
full_total <- colSums(counts)
counts <- counts[keep,,drop=FALSE]
timings <- list()
warnings_seen <- character()
withCallingHandlers({
    tic <- proc.time()[["elapsed"]]
    dds <- DESeqDataSetFromMatrix(counts, meta, design=~condition)
    sizeFactors(dds) <- native_rle
    dds <- DESeq(dds, quiet=TRUE, parallel=FALSE)
    tab <- as.data.frame(results(dds, contrast=c("condition", "case", "control"), alpha=.05))
    tab$gene <- rownames(tab)
    for (alpha in cfg$workflow$q_levels) {
        result <- results(dds, contrast=c("condition", "case", "control"), alpha=alpha)
        tab[[sprintf("q_%.2f", alpha)]] <- result$padj
    }
    tab$p_up <- results(dds, contrast=c("condition", "case", "control"), altHypothesis="greater")$pvalue
    tab$p_down <- results(dds, contrast=c("condition", "case", "control"), altHypothesis="less")$pvalue
    tab$replaced <- if ("replace" %in% names(mcols(dds))) mcols(dds)$replace else FALSE
    tab$beta_converged <- mcols(dds)$betaConv
    tab$maxCooks <- if ("maxCooks" %in% names(mcols(dds))) mcols(dds)$maxCooks else NA_real_
    write_tsv(tab, "DESeq2.tsv")
    write_tsv(data.frame(sample=colnames(dds), size_factor=sizeFactors(dds)), "DESeq2-factors.tsv")
    timings$DESeq2 <- proc.time()[["elapsed"]]-tic
    rm(dds); invisible(gc())
    design <- model.matrix(~condition, meta)
    for (normalization in c("TMM", "RLE")) {
        tic <- proc.time()[["elapsed"]]
        y <- DGEList(counts, group=meta$condition)
        if (normalization == "TMM") {
            y <- calcNormFactors(y, method="TMM")
        } else {
            y$samples$lib.size <- native_rle*exp(mean(log(full_total)))
            y$samples$norm.factors <- 1
        }
        fit <- glmQLFit(y, design=design, robust=TRUE, legacy=FALSE)
        test <- glmQLFTest(fit, coef=match("conditioncase", colnames(design)))
        tab <- topTags(test, n=Inf, sort.by="none")$table
        write_tsv(data.frame(gene=rownames(tab), tab), paste0("edgeR-", normalization, ".tsv"))
        write_tsv(data.frame(sample=rownames(y$samples), y$samples), paste0("edgeR-", normalization, "-factors.tsv"))
        timings[[paste0("edgeR-", normalization)]] <- proc.time()[["elapsed"]]-tic
        rm(y, fit, test); invisible(gc())
    }
}, warning=function(w) {
    warnings_seen <<- c(warnings_seen, conditionMessage(w))
    invokeRestart("muffleWarning")
})
write_json(list(status="complete", genes_eligible=sum(keep), seconds=timings,
                normalization_parity="PASS", warnings=unique(warnings_seen),
                versions=list(R=as.character(getRversion()), DESeq2=as.character(packageVersion("DESeq2")),
                              edgeR=as.character(packageVersion("edgeR")))),
           file.path(root, "native.json"), auto_unbox=TRUE, pretty=TRUE, null="null")
capture.output(sessionInfo(), file=file.path(root, "native-session.txt"))
