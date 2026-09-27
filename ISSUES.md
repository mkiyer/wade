# Open issues

The [release plan](PLAN.md) governs the point-based 0.2.0 validation sequence.
The [kernel roadmap](docs/kernel-validation-roadmap.md) is deferred research. Historical issue
numbers are retained here for continuity, without repeating study reports.
The [full prior discussion](archive/development-20260923/ISSUES.md) is archived.

| Issue | Remaining work |
| --- | --- |
| 29 — Biomarker objective | User rejects unsigned-only discovery. Correct the benchmark target to requested-direction enrichment, magnitude, normal background and patient utility; diagnose directional failures before another broad comparison. Include motivating 5–10 log2 effects. Patient attribution remains descriptive; kernels deferred. |
| 22 — Calibration | The 120-transcriptome development comparison and audit are complete, including joint up/down and unique-gene outcomes. Freeze the selected procedure, error tolerance and adequately replicated fresh confirmation. Twenty null matrices per design do not certify FDR. The fitted-bulk correction remains retired. |
| 26 — Normalization | Validate RLE/internal controls, fitted-factor uncertainty, reference/test overlap and filtering within the selected workflow. No automatic control nomination is selected. |
| 25 — External comparisons | Native DESeq2/edgeR comparison on 120 matched full matrices is complete for a broad distribution target. Correct the directional scorecard first, then replay established inputs. Report mean-DE and upward-subset tasks separately; opposite-direction calls receive no upward credit. |
| 24 — Prostate | Complete gene-count provenance and fusion-oriented interpretation; distinguish missing fusion annotations from negatives and resolve ETV4 interpretation without gene-specific tuning. |
| 19 — Breast | Refresh gene-count/subtype analyses for the final method. Keep PAM50's expression-derived provenance explicit. |
| 20 — Notebooks and review | Rebuild all six notebooks for the final candidate; approve exact results, figures and claims. Existing point-method artifacts are evidence for their recorded versions. |
| 13 — Release | Final distribution contents, citation/release notes, source/wheel/fallback checks, cross-platform CI and publish authorization. |

## Guidance implemented in the current baseline

Issue 18's [before-running recipe](docs/manual.md#before-running) specifies the
independent unit, contrast, eligibility and case-exclusion rule. Recheck it
against the final point API. Issue 5's private RNA100k workbench remains a local
example, not a validated sparse-assay recommendation.

## Later research

Issues 1, 28, 10, 6 and 17 cover transcriptome rarity, sparse/depth-confounded
assays, single-cell/spatial workflows, composition diagnostics and covariates.
They remain outside the immediate bulk-release certification. Their motivations
and retired two-stage issues are retained in the archived discussion.

Issue 29's measurement-kernel component is also deferred: positive zero-count
uncertainty, individual modes, whole-library coherence, justified width/unseen
mass and demonstrated benefit remain unresolved. The derivations and failed
gates stay in the kernel roadmap; none is a prerequisite for the point release.
