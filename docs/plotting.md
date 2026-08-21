# Plotting — an extension, not the method

WADE's job is the statistic. This layer exists because a result is easier to
believe when you can see it, but it is a **convenience on top of the core**,
not part of it, and it is organized so that stays true:

* `import wade` imports **neither** plotly nor matplotlib. Both are imported
  inside the functions that need them, and a test pins that
  (`tests/test_plotting.py`).
* Neither backend is a dependency. `pip install 'wade[plot]'` adds both;
  `wade[plotly]` or `wade[matplotlib]` adds one. Either alone is enough.
* Nothing in `wade.plotting` is read by a statistic, and nothing in the
  statistic depends on it. Deleting this layer would not change a number.

## The results table is the interface

Every number a figure draws is in the result:

```python
res.report()                       # dict of arrays, in the written order
res.to_frame()                     # ... as a polars DataFrame
wade.write_results(res, "out.tsv") # ... on disk, with a JSON manifest
```

**Exporting and plotting elsewhere is a first-class path**, not a fallback. If
you are more fluent in ggplot, `write_results` and then `read_tsv` loses you
nothing — most RNA-seq DE tools stop exactly here, and the columns are named
so that they are usable without this package. What follows is offered for
convenience.

## Architecture: one data layer, thin renderers

```
      WadeResult
          │
          ▼
   ┌─────────────────┐     pure NumPy, no backend import
   │   data layer    │     gene_panels() · volcano_data() · stages_data()
   │  (dataclasses)  │     each with .table()
   └────────┬────────┘
            ├──────────────► plotly renderer
            └──────────────► matplotlib renderer
```

`wade/plotting/` is that diagram, one file per box:

| module | what it holds | what it imports |
|---|---|---|
| `theme.py` | the palette and every label a figure carries | **nothing at all** |
| `data.py` | the three dataclasses and the arrays they hold | NumPy and the core |
| `_plotly.py` | the interactive renderer | plotly, *inside* its functions |
| `_matplotlib.py` | the static renderer | matplotlib, *inside* its functions |
| `__init__.py` | backend resolution and the three `plot_*` | a renderer, on use |

The dependencies only ever point left, and the two renderers never see each
other — which is why a third one is additive rather than surgery.

The split is load-bearing rather than decorative:

* **A figure says the same thing in either backend**, because both read the
  same dataclass. They differ in rendering, never in content.
* **`.table()` on every dataclass** is the chart's table-view twin — the exact
  arrays a panel draws, for a reader who would rather see numbers, or for a
  third renderer.
* **Writing a third renderer is about 100 lines** against the data layer. That
  is the seam to use for altair, bokeh, plotnine, or your lab's house style.

```python
from wade.plotting import volcano_data

d = volcano_data(res, "subset", x="subset_log2_fc", y="z_subset")
d.table()          # gene, x, y, p, padj, significant, colour
d.xlabel, d.ylabel # what the axes mean
```

## The three figures

| function | the question it answers |
|---|---|
| `plot_gene` | *What does this gene's difference look like?* |
| `plot_volcano` | *Which genes?* |
| `plot_stages` | *What kind of difference?* |

**`plot_gene`** is the figure that makes the method legible: the log-ratio
curve `R(p)` above, the two quantile functions it is the ratio of below. A
**flat** curve is a global fold change; a curve that sits at zero and then
**climbs** is a subset. The dashed line is `log2` of the *fitted* global fold
change — the shift the subset test takes as its null — so the curve's
departure from it is exactly what `p_subset` prices.

**`plot_volcano`** is effect size against significance, coloured by
`affected_fraction` so the *shape* of each difference is visible in the
overview. `stage="both"` puts the two stages side by side on a shared x axis,
which is the view where a subset gene sits low on the first panel and high on
the second.

**`plot_stages`** is the two stages against each other — the reading table in
the README, drawn, with each gene in the quadrant that names its pattern.

### Either axis takes any column

This matters at scale. On a large cohort the p-values saturate at the
resolution floor, and a volcano's y axis becomes a flat line of ties:

```python
# the view that still separates genes when p-values have stopped
plot_volcano(res, stage="subset", x="subset_log2_fc", y="z_subset", label=8)
```

`x=` and `y=` name any column of `res.columns()`; `color=` takes a column name
or your own per-gene array (a gene set, a cluster, a QC score). Off the
p-value axis the BH cutoff line is suppressed, because it means nothing there.
`label=n` names the top *n* by y, **broken by |x|**, so ties at the floor
select the largest effects rather than an arbitrary handful.

## Per-gene metadata is displayed, never used

WADE's core needs **one gene id column and nothing more**. If your input frame
carries other per-gene columns — a symbol, a biotype, a chromosome — they are
carried alongside and available to figures and tables for convenience. No
statistic reads them, and dropping them changes no number. Gene symbols are
the case that matters most: a ranked table of `ENSG…` accessions is unreadable,
and one of symbols is biology.

## What this layer is not for

* **Publication typesetting.** Fonts, panel letters, journal templates —
  export the table.
* **Arbitrary layouts.** Three figures answer three questions; a fourth
  question is a `.table()` and your own code.
* **Anything the statistic depends on.** If a change here would alter a
  reported number, it belongs in the core, not here.

## Backends

| | plotly | matplotlib |
|---|---|---|
| role | the interactive default when installed | the static / publication backend |
| hover carrying every column | yes | — |
| large clouds | `Scattergl` past 2,000 points | rasterized as needed |
| static export | needs `python-kaleido` **and** a Chrome/Chromium | PDF/SVG with nothing else |

`backend="plotly"` or `"matplotlib"` picks one explicitly;
`available_backends()` reports what imports in this environment.
