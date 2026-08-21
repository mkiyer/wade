# Plan: finish the reorganization, then the visualization extension

A working document. **Delete it when Phase C lands** — its outcome belongs in
`ROADMAP.md`, `docs/plotting.md` and the code. Written 2026-08-21.

## Status

| Phase | State |
|---|---|
| **A — make the documents true** | **done** (`340fc1f`) |
| **D — the last six core gaps** | **done** (`46e9446`) |
| **B — plotting becomes a subpackage** | **next** |
| **C — the visualization work** | after B |
| Notebooks (three of them) | after C |

Order decided 2026-08-21: documents true first, then the core finished, then
the reorganization — so the split happens against a codebase with nothing
outstanding in it. That is now the case: `pytest -q` is **694 passing in ~10 s**,
parity at **9.155e-15** over 436 comparisons, working tree clean.

**Standing constraints for everything below.** Nothing may change a reported
number (`pytest -q` is the check, and the parity ledger printed at the end of
the run is the sharper one). Fast paths that *do* change numbers are opt-in and
named. Measure before and after, and put the numbers in the doc that owns the
subject, not in a commit message. Prefer deleting to adding: the goal is clean,
concise, readable, maintainable, simple, elegant.

---

## Phase B — plotting becomes a subpackage

`src/wade/plotting.py` is **1,124 lines, the largest module in the package**,
bigger than `api.py` (931). It is already lazily imported and already has a
data-layer/renderer split *described in comments*; B makes that structural.

### B1. Split the module

The seams are the file's own section banners, which are stable landmarks even
as line numbers move:

| New file | Take the section(s) | Contents |
|---|---|---|
| `plotting/theme.py` | `# Theme` | `_INK`…`_DIVERGING`, `_COLOR_SPECS`, `_STAGE_LABELS`, `_AXIS_LABELS`, `_axis_label`, `_column_array`. **This is where C1's tokens land.** |
| `plotting/data.py` | `# Data layer: the single-gene panel`, `# Data layer: the point clouds` | `GenePanel`/`gene_panels`, `VolcanoData`/`volcano_data`, `StagesData`/`stages_data` and their private helpers (`_bh_cutoff`, `_neglog10`, `_stage_arrays`, `_color_arrays`, `_result_columns`, `_label_mask`, `_fmt_p`, `_panel_from_detail`, `_stats_for`, `_name_or`). **Imports no backend.** |
| `plotting/_plotly.py` | `# plotly renderers` | `_plotly_layout`, `_gene_plotly`, `_volcano_plotly`, `_stages_plotly` and their helpers. |
| `plotting/_matplotlib.py` | `# matplotlib renderers` | the matplotlib twins. |
| `plotting/__init__.py` | `# Backend resolution`, `# Public API` | `BACKENDS`, `DEFAULT_BACKEND`, `available_backends`, `_resolve_backend`, `plot_gene`, `plot_volcano`, `plot_stages`, and re-exports of everything in `data.py` and the two dataclasses' friends. `__all__` **unchanged**. |

Rules for the move:

* **No public name changes and no behaviour changes.** The diff should be
  almost entirely relocation. If something needs rewriting to move cleanly,
  that is a signal the seam is wrong — move it as-is and note it.
* The renderers are imported **inside** the `plot_*` functions, exactly as
  today. `plotting/__init__.py` must not import plotly or matplotlib at module
  scope, or `import wade` stops being NumPy-only.
* Private helpers travel with their only caller. Anything used by both the
  data layer and a renderer goes in `theme.py` (it will be a label or a token).

*Verify:*

```bash
pytest -q                          # 694, unchanged
pytest -q tests/test_plotting.py   # 33 collected, unchanged
python -c "import sys, wade; assert not {'plotly','matplotlib'} & set(sys.modules)"
git diff --stat                    # should read as a move, not a rewrite
```

### B2. Make the data layer first-class

* `GenePanel` gains the `.table()` its two siblings already have — the numbers
  behind the flagship figure, currently extractable only by hand.
* One test that builds all three dataclasses and calls `.table()` on each,
  asserting the arrays match the result's own columns.
* `docs/plotting.md` already documents this seam; check its claims still hold
  after the move and update the one code block if the import path shifts.

---

## Phase C — the visualization work

Independent items; land them separately. Each is cheaper after B.

### C1. Theme and palette tokens — do this first

Everything else in C renders through it.

* One dataclass of tokens (ink, muted, grid, axis, surface, case, control,
  sequential scale, diverging scale, font), three built-ins: `"light"`
  (today's), `"dark"`, `"high-contrast"`.
* `theme=` on `plot_gene` / `plot_volcano` / `plot_stages`, accepting a name or
  a token instance; a module-level default for a whole session.
* plotly's hard-coded `plotly_white` template must follow the theme, or dark
  mode is a light slab with dark points.
* *Verify:* every built-in renders all three figures on both backends; a test
  asserts no colour literal remains outside `theme.py`.

### C2. Draw the bootstrap intervals

All five descriptors carry one and no figure shows any, so a subset resting on
four affected samples looks exactly as firm as one resting on four hundred.
Error bars on whichever axis holds a descriptor; a band in `plot_gene`. Draw
nothing when `n_boot=0`.

### C3. A driver figure

The figure form of `subset_drivers` + `library_qc`: which samples are in the
affected region, with their complexity and depth beside them. This is the check
that reversed a tempting Ewing-sarcoma reading of the real data
(`scaling.md` §7.2) and it should not be text-only. Show the gene's share of
each driver's library — that is the number that separated the artefact from
`ETV4`.

### C4. Optional per-gene metadata in figures and tables

`WadeResult.gene_meta` is carried and read by nothing (Phase D). Let the
display layer show it: gene symbol as the point label and in the hover, other
columns on request. **No statistic may read it** — a test already asserts that
adding metadata moves no number; keep that true.

### C5. Linked views (plotly)

Click a volcano point, see that gene's panel. `go.FigureWidget` callback over
the existing data layer. **Document the limitation**: needs a live kernel, so
it does not survive into exported HTML.

### C6. A stage-1 figure

`GeneDetail.cumulative_area` is stage 1's statistic and is drawn nowhere.

### C7. Label de-collision

Last, because C1/C2 change the geometry. The smear is worst where p-values tie,
which is where real cohorts sit. `adjustText` only if an in-house approach
fails — it would be a third optional dependency.

---

## Then: notebooks

Three deliverables, in this order, after C.

1. **`demo`** — small synthetic data, planted ground truth. The showcase and
   gallery: a README in executable form, every claim checkable. Its job is to
   answer "what does this tool do?" in one read.
2. **`benchmark`** — the head-to-head, on simulated counts *and* a few real
   datasets from the literature: COPA / OS / ORT / MOST / LSOSS as the
   subset-detection competitors, t-test and Wilcoxon as floors, waddR as the
   nearest Wasserstein relative. Where WADE's claim is tested against
   alternatives rather than against itself.
3. **`rna100k`** (exists) — manuscript material, not a demo. Relabel it as
   such.

---

## Not doing, and why

Mirrored in `ROADMAP.md` §4 so it survives this document's deletion: a second
chunked driver, `cpm`/`rle` as functions, format readers before the data is in
hand, gene metadata in the statistics, `adjustText` by default, publication
typesetting.
