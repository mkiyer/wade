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

The one exception is `plot_drivers`, which is per-**sample** rather than
per-gene: its numbers come from `wade.subset_drivers` and `wade.library_qc`,
both public and both callable without this layer.

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

## The four figures

| function | the question it answers |
|---|---|
| `plot_gene` | *What does this gene's difference look like?* |
| `plot_volcano` | *Which genes?* |
| `plot_stages` | *What kind of difference?* |
| `plot_drivers` | *Should I believe this one?* |

`plot_linked` is a fifth function rather than a fifth question: it is the
volcano and the gene panel in one live widget, and it has its own section
below.

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

**`plot_drivers`** is the check to run before believing a hit, and the one that
reversed a tempting Ewing-sarcoma reading of a real plasma cohort
(`docs/scaling.md` §7.2). A distributional test faithfully reports a subset in
every gene that a low-complexity library happens to detect, so *which* samples
are in the affected region is only half the answer; whether those samples are
unremarkable is the other half. Four columns, one row per driver, each read
against the **cohort** — a dashed line at its median, a band over its
interquartile range — because "5,563 genes detected" means nothing except
beside "11,114 elsewhere":

| column | what it tells you |
|---|---|
| the gene, normalized | why these samples |
| **its share of the library** | **the discriminating number** |
| complexity | the fraction of genes the library detects |
| library size | its depth in counts |

```python
plot_drivers(res, "ETV4", counts)      # the counts the run was given
```

`counts` is required and is the raw matrix: a `WadeResult` does not carry it —
`res.tpm` is normalized *and* jittered, so it has no exact zeros left and
cannot be asked what a library detected — and WADE does not read files, so the
caller who has the counts passes them. Drivers that are ordinary libraries
which happen to share a diagnosis are the finding; drivers that are
low-complexity outliers are the artefact. A gene taking 12% of a driver's
library is telling you about the library; one taking a fraction of a percent is
telling you about the gene. `DriverPanel.table()` is the same thing as numbers,
and carries the libraries' concentration, which the figure does not draw.

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

### Stage 1's statistic, accumulating

`plot_gene(..., cumulative_area=True)` adds a middle row holding
`GeneDetail.cumulative_area`, which starts at zero and whose **last point is
`mean_shift`**. Where it gets there from is the reading:

* a straight ramp is a global shift, contributed evenly across the
  distribution;
* a flat line that turns up only near `p = 1` is a mean shift produced almost
  entirely by a few samples — and it turns up exactly where the log-ratio curve
  above it leaves the dashed reference.

That is the two stages agreeing on one gene, in one figure. Measured on the
README's dataset: at the halfway quantile the global 2× gene has accumulated
39% of its mean shift, and the 15% and 5% subset genes 3.0% and 4.0%.

It is off by default. The log-ratio curve is what makes the method legible and
a third row is a diagnostic, not the headline.

### Bootstrap intervals are drawn, not just carried

All five descriptors carry a 95% interval at `n_boot > 0`, and until now no
figure showed any — so a subset resting on four affected samples looked exactly
as firm as one resting on four hundred. Two places now show them, and **nothing
is drawn at `n_boot = 0`**, because a figure must not imply a precision the run
did not measure.

* **On a volcano**, error bars on whichever axis holds a descriptor the result
  has an interval for. The rule is the column's name: an axis holding `foo`
  gets bars iff the result carries `ci_foo`, so `x="subset_log2_fc"` and
  `y="affected_fraction"` both get them and a p-value or `z_subset` axis gets
  none. Bars go on the **labelled** points only — a transcriptome of error bars
  is mush, `label=` is already the caller saying which genes matter, and every
  gene's interval is in `res.columns()` and therefore in the hover.
* **On `plot_gene`**, a band marking where the **edge of the affected region**
  lies. `affected_fraction` is a participation ratio — for a subset of size
  `pi` it reads `pi` — so it is an extent along the quantile axis: the top `pi`
  when the change is upward, the bottom `pi` when downward. The band is that
  edge's interval, and it lands exactly where the curve leaves the dashed
  reference. A narrow band is a well-determined extent; a wide one is a subset
  resting on a handful of samples.

## Per-gene metadata is displayed, never used

WADE's core needs **one gene id column and nothing more**. If your input frame
carries other per-gene columns — a symbol, a biotype, a chromosome — they are
carried on `res.gene_meta` and shown on request. No statistic reads them, and
dropping them changes no number; a test asserts exactly that, on the way in and
again at this layer.

```python
plot_volcano(res, label=8, meta="gene_name")              # symbols, not ENSG…
plot_volcano(res, label=8, meta=["gene_name", "gene_type"])
```

`meta=` names one column or several. **The first names the points** — the panel
titles in `plot_gene`, the point label and hover title on a cloud — because a
ranked cloud of `ENSG…` accessions is unreadable and one of symbols is biology.
Every named column joins plotly's hover and every `.table()`. The gene id stays
on the dataclass as `gene`, so the table remains keyed by the id the core used.

One consequence worth knowing: with `meta=` set, `label=` matches these names
too, so `label=["MYC"]` finds MYC. You name what you see.

## What this layer is not for

* **Publication typesetting.** Fonts, panel letters, journal templates —
  export the table.
* **Arbitrary layouts.** Four figures answer four questions; a fifth question
  is a `.table()` and your own code.
* **Anything the statistic depends on.** If a change here would alter a
  reported number, it belongs in the core, not here.

### Labels are placed, not just offset

`label=n` names the top *n* genes, and on a real cohort those *n* land in a
tight cluster — thousands of genes tie at the p-value floor, so the most
significant ones share a position and their text stacks into a smear. Drawing
the bootstrap intervals made it worse, by putting error bars through it.

Labels are now placed by a greedy slot assignment: most significant first, each
taking the first candidate position that clears every label already placed.
It runs in normalized axis units in the data layer, so **both backends place
text identically** and neither needs font metrics, and displacement is capped at
a couple of label-heights — there are no leader lines, so a label has to stay
next to its point.

Overlapping label pairs on the README's dataset, alternate-above-and-below
against the slot assignment:

| figure | labels | before | after |
|---|---|---|---|
| volcano, mean-shift stage | 8 | 24 | **1** |
| volcano, subset stage | 8 | 28 | **0** |
| two-stage plot | 8 | 24 | **0** |
| subset magnitude vs affected fraction | 12 | 25 | **3** |

The last row is the honest limit: twelve labels on a cluster of near-coincident
points need more width than the cluster has, and no placement rule invents
space. Label fewer, or read the rest off the hover — `adjustText` was not
brought in for it, and would not have fixed that row either.

## Themes

Every figure takes `theme=`, and three are built in:

| | `"light"` | `"dark"` | `"high-contrast"` |
|---|---|---|---|
| ground / ink | white / near-black | near-black / off-white | white / pure black |
| case, control | blue, orange | brightened blue, orange | strong blue, strong red-orange |
| sequential ramp | `viridis` | `viridis` | `cividis` |
| plotly template | `plotly_white` | `plotly_dark` | `plotly_white` |

```python
plot_volcano(res, theme="dark")

import wade.plotting
wade.plotting.DEFAULT_THEME = "dark"     # ... or for the whole session

from dataclasses import replace          # ... or your lab's palette
house = replace(wade.plotting.THEMES["light"], case="#7b3fa0", ctrl="#3fa07b")
plot_stages(res, theme=house)
```

`"light"` is the default and is what the figures above were drawn with;
switching to it renders **pixel-identically** to the palette that preceded
themes, which is how the token set was checked. `"high-contrast"` is for a
projector or a reader with colour-vision deficiency: `cividis` is built for
CVD where `viridis` is only safe for it.

Two rules keep this honest, and both are pinned by tests:

* **No colour literal lives outside `theme.py`.** A hex string written into a
  renderer is one no other theme can follow — which is exactly how a dark
  figure becomes a light slab with dark points. That includes plotly's
  template and the boxes behind in-plot labels.
* **The data layer names a colour *role*, never a colormap.** `volcano_data`
  says `affected_fraction` is sequential and `direction` is diverging; the
  theme says what sequential and diverging look like. So the meaning of a
  colour is fixed across themes — dark at 0 (a subset), light at 1
  (everything); blue down, red up — while its palette is not.

## Linked views: click a point, see the gene

`plot_linked` puts the two views that answer each other in one figure — the
volcano and, beside it, whichever gene you last clicked:

```python
plot_linked(res, "subset", label=8)     # then click
```

**Its limitation is real and is not a bug.** It returns a plotly
`FigureWidget`, and a widget is a live object: the click handler runs in *your
Python kernel*. So it works in Jupyter, JupyterLab, VS Code and Colab for as
long as that kernel is alive — and it **does not survive export**. Saved to
HTML, or reopened from a notebook whose kernel has stopped, it is a static
picture of whichever gene was last drawn. Nothing is lost, but nothing is
linked either. For a figure that must travel, use `plot_volcano` and
`plot_gene` separately; they draw the same arrays from the same data layer.

It is plotly-only and needs `anywidget` (`conda install anywidget`, or
`pip install 'wade[linked]'`). Every static figure works without it.

## Backends

| | plotly | matplotlib |
|---|---|---|
| role | the interactive default when installed | the static / publication backend |
| hover carrying every column | yes | — |
| large clouds | `Scattergl` past 2,000 points | rasterized as needed |
| static export | needs `python-kaleido` **and** a Chrome/Chromium | PDF/SVG with nothing else |

`backend="plotly"` or `"matplotlib"` picks one explicitly;
`available_backends()` reports what imports in this environment.
