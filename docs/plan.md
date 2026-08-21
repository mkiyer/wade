# Plan: make the documents true, and make visualization an extension

A working document, for review before execution. **It is deleted when done** —
its outcome lives in `ROADMAP.md`, `docs/plotting.md` and the code. Written
2026-08-21, after the correctness and simplification passes landed
(`fddf2c1` … `d2bddee`).

Two goals, in this order:

1. **The documents do not describe the package any more.** Fix that first,
   because a stale queue makes every later decision worse.
2. **Visualization should read as an extension of the statistical core**, not
   as part of it — decoupled by organization, documentation and demos.

**Decided 2026-08-21** (§7's questions, answered):

* **Order: A → D → B**, then C in a later session. Documents true first, then
  the core finished, then the reorganization — so the subpackage split happens
  against a codebase with nothing outstanding in it.
* **B1 goes ahead**, with organization left to judgement. The bar: clean,
  concise, readable, maintainable, simple, elegant.
* **D6 is not a data-boundary change.** WADE needs *one gene id column and
  nothing more*. Extra per-gene metadata is a **display** concern: carry it if
  the user supplied it, and let the display layer optionally show it (gene
  symbol above all). That moves the work out of the core and into Phase C,
  and shrinks it.
* **Notebooks are three separate deliverables, all after the implementation**
  — see §8.

---

## 0. The problem, measured

`ROADMAP.md` is 301 lines and much of it is wrong or duplicated:

| Section | Lines | Problem |
|---|---|---|
| §1 Demo notebook | 24 | Still says "synthetic data, no download, decided" — but `notebooks/rna100k.qmd` exists and runs on the real cohort. Two different deliverables are conflated. |
| §2 What the real data demands | 95 | Mostly **scale narrative and measurements**, which is `docs/scaling.md`'s job. A queue should say what to do, not re-derive why. |
| §3 The audited queue | 56 | Raw audit output. **Eleven of its items are already fixed**; two describe a function that no longer exists; the prose is auditor-voice, not queue-voice. |
| §4 Inference refinements | 28 | Broadly accurate. |
| §5 Format helpers, other small things | 27 | **Duplicates §3** on the plotting items and the format helpers; says pandas is untested (it is tested now). |
| §6 Packaging and release | 7 | Says CI is unstarted. CI exists (`.github/workflows/ci.yml`). |

And `src/wade/plotting.py` is **1,124 lines — the largest module in the
package**, bigger than `api.py` (931). It is already lazily imported and
already has a documented data-layer/renderer split, but nothing in the file
layout, the export list or the docs says "extension".

Verification for this section: `wc -l`, and the eleven fixed items are each
traceable to a commit in `git log fddf2c1..d2bddee`.

---

## 1. Target document structure

| File | Owns | Now | Target |
|---|---|---|---|
| `README.md` | The user-facing contract | 354 | ~330 (plotting section → pointer) |
| `ROADMAP.md` | **A work queue. Nothing else.** | 301 | **~110** |
| `docs/method.md` | What WADE computes and why | 797 | unchanged |
| `docs/limits.md` | What it cannot do | 260 | unchanged |
| `docs/scaling.md` | Large-data agenda + every measurement | 667 | ~700 (absorbs ROADMAP §2's narrative) |
| `docs/plotting.md` | **New.** The visualization extension | — | ~120 |
| `docs/implementation-notes.md` | Parity, kernels, cross-language traps | 357 | unchanged |
| `docs/HANDOFF.md` | What is not obvious from the above | 429 | ~430 (map updated) |

The rule that decides where a paragraph goes: **ROADMAP says what to do next;
everything explaining *why* lives in the doc that owns the subject.**

---

## 2. Phase A — make the documents true

No code changes. Each step is verified by reading the result, plus the greps
named.

**A1. Rewrite `ROADMAP.md` as a queue.** Target ~110 lines, structured as:

- *Preamble*: scope (counts only), the standing decisions, and what landed
  recently with commit references rather than re-explanation.
- *§1 Now* — the two or three things actually next, each ≤ 5 lines.
- *§2 Soon* — real items with a use case, grouped, one line each.
- *§3 Not doing, and why* — the list that stops items reappearing. This
  section is load-bearing: it is what keeps the queue from re-growing.
- *§4 Open questions* — genuine unknowns, unchanged in spirit.

Drop from §3 the eleven items now fixed: the `cumulative_area` invariant, the
`slow` marker, the README dashed line, the bare-`pytest` step, `mamba_env.yaml`
drift, manifest provenance, `alternative="less"`, pandas, `allow_extra`,
plotly's shared x axis, and `subset_log2_fc`'s missing interval. Drop the two
`wade_from_matrix` items — the function is gone. Merge §5's plotting bullets
into the visualization group and delete §5. Rewrite §6 to name CI as done and
wheels/publishing as release-time.

*Verify:* `grep -c` for each dropped item returns 0; `wc -l ROADMAP.md` ≈ 110.

**A2. Move ROADMAP §2's narrative into `docs/scaling.md`.** The measurements
(depth imbalance 1.7×, 18.8% of genes at the floor, 47% zeros, the artefact
libraries) belong beside the other measured findings. ROADMAP keeps a ~12-line
§1 entry pointing at them.

*Verify:* every number in the moved text appears exactly once in the repo.

**A3. Write `docs/plotting.md`.** The extension's own document:

- The stance: **the results table is the interface.** Every statistic is in
  `res.report()` / `write_results()`; plotting is a convenience, and exporting
  to ggplot or seaborn is a first-class path, not a fallback.
- The architecture: one pure-NumPy data layer (`gene_panels`, `volcano_data`,
  `stages_data`, each with `.table()`), two thin renderers. Writing a third
  renderer is ~100 lines against the data layer.
- The three figures, what question each answers, and how to re-axis them
  (`x=`, `y=`, `color=`).
- What is *not* WADE's job: publication typesetting, arbitrary layouts.
- The theme/palette contract, once B1 lands.

**A4. README plotting section → pointer**, and update the documentation map in
`README.md` and `docs/HANDOFF.md` to list `docs/plotting.md`.

---

## 3. Phase B — organization: plotting becomes a subpackage

**B1. Split `src/wade/plotting.py` (1,124 lines) into `src/wade/plotting/`.**
The seams already exist in the file's own section comments:

| New file | From (current lines) | Holds |
|---|---|---|
| `__init__.py` | 133–173, 624–740 | backend resolution (`available_backends`, `_resolve_backend`), the three `plot_*` dispatchers, and the data layer re-exported |
| `theme.py` | 75–131 | colour tokens, fonts, axis labels, the column resolver — **and where dark mode and the palette knob land** |
| `data.py` | 175–622 | `GenePanel`, `VolcanoData`, `StagesData` and their builders. Pure NumPy, no backend import |
| `_plotly.py` | 742–… | the plotly renderer |
| `_matplotlib.py` | remainder | the matplotlib renderer |

Why this and not "leave it and document it": the theme work in Phase C touches
one file instead of grepping a 1,124-line module, the data layer's
independence becomes structural instead of a comment, and the largest module
in the package stops being the one that is not the method.

*Verify:* `tests/test_plotting.py` passes unchanged — 27 test functions,
33 collected with parametrization — because they import
`wade.plotting as P`, which a subpackage still resolves. `import wade` still imports
neither plotly nor matplotlib (already pinned by
`test_importing_wade_imports_neither_frame_library` and
`test_importing_wade_costs_only_numpy`); no public name changes, so the
diff is a move.

**B2. Export the data layer.** `volcano_data`, `stages_data`, `gene_panels`,
`GenePanel`, `VolcanoData`, `StagesData` become importable from
`wade.plotting` explicitly, and `GenePanel` gains the `.table()` its two
siblings have. This is the "plot it yourself" seam made real.

*Verify:* a test that builds all three dataclasses and calls `.table()` on
each with no backend installed.

---

## 4. Phase C — the visualization work

Each item is independent and lands separately. Phase B first makes each
cheaper.

| Item | Size | Note |
|---|---|---|
| **C1. Theme + palette tokens** | medium | One token set, three built-ins (light / dark / high-contrast), `theme=` on every plot function and a module default. Dark mode is then a palette, not a feature. |
| **C2. Draw the bootstrap intervals** | medium | Error bars on the volcano's y (or x, when the axis is a magnitude), a band in `plot_gene`. All five descriptors now carry intervals, so there is something to draw for whatever axis is selected. |
| **C3. Driver figure for a gene** | medium | The figure form of `subset_drivers` + `library_qc`: which samples are in the affected region, and their complexity/depth beside them. This is the check that reversed the Ewing reading; it should not be text-only. |
| **C4. Linked views (plotly)** | medium | Click a volcano point → its `plot_gene` panel. `go.FigureWidget` callback. **Caveat to document: needs a live kernel, so it does not survive into exported HTML.** |
| **C5. A stage-1 figure** | small | `GeneDetail.cumulative_area` is stage 1's statistic and is drawn nowhere. |
| **C6. Label de-collision** | small | Only worth doing after C1/C2; the current smear is worst exactly where p-values tie. |

---

## 5. Phase D — the remaining core items

Short, because the audit's correctness half is done. Ordered by whether a use
case exists today.

| Item | Size | Use case |
|---|---|---|
| **D1. Stage-2 exceedance / refinement flags** | small | `limits.md` §5 makes a reading rule out of "is this gene *at* the floor" and it is only appliable to stage 1. `nexc`/`refined` are computed for stage 2 and thrown away — this is plumbing, not new work. |
| **D2. Export the GPD-extrapolation flag** | small | Same fix; `refined_*` should reach the written table so a reader can tell a counted p-value from an extrapolated one. |
| **D3. Combinatorial-floor helper** | small | The one check `limits.md` says to do *before* running. `permutation_space()` already does the restricted case; this is the `C(n1,k)/C(n,k)` sibling. |
| **D4. Bootstrap `level=`** | small | Settable from `wade()`; `characterization_ci` already takes it. |
| **D5. Missing-count errors say where** | small | Name the row/column instead of just the condition. |
| **D6. Carry per-gene metadata (do not use it)** | small | `as_counts` keeps *numeric* leftover columns in `Counts.meta` and silently **drops string ones**, so a frame of `gene_id, gene_name, samples…` loses the symbol. WADE's core needs one id and nothing more, so this is only about *carrying*: keep every per-gene column, expose it on the result, and let the display layer optionally show it (Phase C). No statistic reads it. |

---

## 6. Not doing, and why

To be recorded in the rewritten ROADMAP so it stops resurfacing:

- **A second chunked driver** for pre-normalized input — there is no
  pre-normalized entry point any more.
- **An API documentation build** — deferred to release (Phase E), not
  abandoned. Twelve modules of essay-length docstrings; Sphinx earns its
  keep when there is a published version to point at.
- **featureCounts / MatrixMarket readers** — built when the data is in hand
  and the shape is known, per the standing "polars can read it" decision.
- **Drawing `GeneDetail.cumulative_area`** — superseded by C5, which does it
  properly as a stage-1 figure rather than an extra line.
- **`adjustText` as a dependency** — reconsider only if C6's own approach
  fails.

---

## 7. Decisions I need from you

1. **B1, the subpackage split.** It is a pure move (no public name changes,
   tests unchanged), but it touches the largest file in the package. Do it, or
   leave `plotting.py` as one module and decouple by docs alone?
2. **D6.** Widen the data boundary to carry string annotation columns, or
   document the two-line polars join and close the item?
3. **Phase order.** A → B → C → D as written, or pull D1–D3 (small, core,
   correctness-adjacent) ahead of the visualization work?
*(All four resolved — see the decisions at the top and §8.)*

---

## 8. Notebooks — three deliverables, not one

Sequenced **after** all of the above, because each documents behaviour that
should have stopped moving first.

| Notebook | Data | Purpose |
|---|---|---|
| **`demo`** | small, synthetic, in-notebook | The showcase and gallery: a README in executable form. What the tool does, what each figure answers, planted ground truth so every claim is checkable. |
| **`benchmark`** | simulated counts **plus a few real datasets from the literature** | The head-to-head: COPA / OS / ORT / MOST / LSOSS as the subset-detection competitors, t-test and Wilcoxon as floors, waddR as the nearest Wasserstein relative. This is where WADE's claim is tested against alternatives rather than against itself. |
| **`rna100k`** (exists) | the real 31k × 83k cohort | Not a demo — it is manuscript material. Keep it as the real-data workbench and say so. |
