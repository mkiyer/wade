# Open issues

The [canonical scientific plan](docs/kernel-validation-roadmap.md) governs the
research sequence; [PLAN.md](PLAN.md) is the release queue. Historical issue
numbers are retained here for continuity, without repeating study reports.
The [full prior discussion](archive/development-20260923/ISSUES.md) is archived.

| Issue | Remaining work |
| --- | --- |
| 29 — Biomarker objective and kernels | Gate-B instruments and depth-null replay are implemented. The upper walk fails as a general prevalence estimator even without counting noise. Complete the remaining mechanism replays and validate one measurement-kernel procedure; current Beta candidates remain unselected. |
| 22 — Calibration | Validate the selected complete kernel workflow at transcriptome scale. The two-stage fitted-bulk correction is retired; reopening it is not the next task. |
| 26 — Normalization | Validate RLE/internal controls, fitted-factor uncertainty, reference/test overlap and filtering within the selected workflow. No automatic control nomination is selected. |
| 25 — External comparisons | Compare with native DESeq2/edgeR and separately defined OUTRIDER/related outlier tasks. Evaluate effects and subsets as well as detection. |
| 24 — Prostate | Complete gene-count provenance and fusion-oriented interpretation; distinguish missing fusion annotations from negatives and resolve ETV4 interpretation without gene-specific tuning. |
| 19 — Breast | Refresh gene-count/subtype analyses for the final method. Keep PAM50's expression-derived provenance explicit. |
| 20 — Notebooks and review | Rebuild all six notebooks for the final candidate; approve exact results, figures and claims. Existing point-method artifacts are evidence for their recorded versions. |
| 13 — Release | Final distribution contents, citation/release notes, source/wheel/fallback checks, cross-platform CI and publish authorization. |

## Guidance implemented in the current baseline

Issue 18's [before-running recipe](docs/manual.md#before-running) specifies the
independent unit, contrast, eligibility and case-exclusion rule. Recheck it
against the final kernel API. Issue 5's private RNA100k workbench remains a local
example, not a validated sparse-assay recommendation.

## Later research

Issues 1, 28, 10, 6 and 17 cover transcriptome rarity, sparse/depth-confounded
assays, single-cell/spatial workflows, composition diagnostics and covariates.
They remain outside the immediate bulk-release certification. Their motivations
and retired two-stage issues are retained in the archived discussion.
