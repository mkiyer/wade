"""The plotly renderer: interactive figures, one function per public figure.

``import plotly`` happens inside each renderer, not here, so that importing
this module costs nothing and ``import wade`` stays NumPy-only. ``go`` is
passed down to the trace helpers for the same reason.
"""

from __future__ import annotations

import numpy as np

from .data import DRIVER_PANELS, _label_positions
from .theme import QUADRANTS, _STAGE_LABELS


def _plotly_layout(fig, theme, *, title, width, height, legend_below=False):
    fig.update_layout(
        template=theme.plotly_template, title=title, width=width, height=height,
        font=dict(family=theme.font, size=12, color=theme.ink),
        paper_bgcolor=theme.surface, plot_bgcolor=theme.surface,
        margin=dict(l=70, r=30, t=70 if title else 50, b=60),
        hoverlabel=dict(font=dict(family=theme.font)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=theme.grid, gridwidth=1, zeroline=False,
                     linecolor=theme.axis, ticks="outside", tickcolor=theme.axis, title_font=dict(color=theme.ink),
                     tickfont=dict(color=theme.muted))
    fig.update_yaxes(showgrid=True, gridcolor=theme.grid, gridwidth=1, zeroline=False,
                     linecolor=theme.axis, ticks="outside", tickcolor=theme.axis, title_font=dict(color=theme.ink),
                     tickfont=dict(color=theme.muted))
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, x=0,
                                      font=dict(color=theme.ink)))
    return fig


def _gene_traces(panel, theme, go, *, showlegend):
    """The three lines a gene panel is: the log-ratio curve, and the two
    quantile functions. Extracted because the linked view rewrites exactly
    these in place, and the two figures must draw the same thing."""
    d = panel.detail
    return (
        go.Scatter(x=d.p, y=d.r, mode="lines", line=dict(color=theme.ink, width=1.8),
                   name="log₂ ratio", legendgroup="r", showlegend=showlegend,
                   customdata=np.c_[d.y1, d.y0],
                   hovertemplate=("p = %{x:.3f}<br>log₂ ratio = %{y:.2f}<br>"
                                  "Q<sub>case</sub> = %{customdata[0]:.3g}<br>"
                                  "Q<sub>ctrl</sub> = %{customdata[1]:.3g}<extra></extra>")),
        go.Scatter(x=d.p, y=d.y1, mode="lines", line=dict(color=theme.case, width=1.8),
                   name="case", legendgroup="case", showlegend=showlegend,
                   hovertemplate="p = %{x:.3f}<br>case Q = %{y:.3g}<extra></extra>"),
        go.Scatter(x=d.p, y=d.y0, mode="lines", line=dict(color=theme.ctrl, width=1.8),
                   name="control", legendgroup="ctrl", showlegend=showlegend,
                   hovertemplate="p = %{x:.3f}<br>control Q = %{y:.3g}<extra></extra>"),
    )


def _gene_plotly(panels, theme, *, share_y, cumulative_area, title, width, height):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = len(panels)
    titles = []
    for p in panels:
        sub = "<br>".join(p.subtitle_lines)
        titles.append(f"<b>{p.name}</b><br><span style='font-size:10px;color:{theme.muted}'>{sub}</span>")
    # Stage 1's statistic slots between stage 2's curve and the raw evidence, so
    # the two statistics sit together and the quantile functions stay last.
    area_row = 2 if cumulative_area else None
    q_row = 3 if cumulative_area else 2
    rows = q_row
    heights = [0.42, 0.29, 0.29] if cumulative_area else [0.55, 0.45]
    fig = make_subplots(rows=rows, cols=k, shared_xaxes=True, shared_yaxes=False,
                        subplot_titles=titles, row_heights=heights,
                        vertical_spacing=0.09 if cumulative_area else 0.12,
                        horizontal_spacing=0.06 if k > 1 else 0.05)
    if share_y:
        # Only the log-ratio row shares a scale: magnitudes of R are comparable
        # across genes, raw expression levels are not.
        for j in range(2, k + 1):
            fig.update_yaxes(matches="y", row=1, col=j)
    for j, p in enumerate(panels, start=1):
        d = p.detail
        ratio, case, ctrl = _gene_traces(p, theme, go, showlegend=(j == 1))
        fig.add_trace(ratio, row=1, col=j)
        fig.add_hline(y=0.0, line=dict(color=theme.axis, width=1), row=1, col=j)
        fig.add_hline(y=p.reference, line=dict(color=theme.muted, width=1, dash="dash"), row=1, col=j)
        # How firm the extent of the affected region is. Added after the trace:
        # plotly skips shapes on subplots that are still empty.
        span = p.affected_span
        if span is not None:
            fig.add_vrect(x0=span[0], x1=span[1], fillcolor=theme.case, opacity=0.16,
                          line_width=0, layer="below", row=1, col=j)
        if area_row is not None:
            fig.add_trace(go.Scatter(
                x=d.p, y=d.cumulative_area, mode="lines",
                # Ink, like the log-ratio curve: both rows draw a statistic, and
                # the y axis names which. No legend entry — one curve in the row.
                line=dict(color=theme.ink, width=1.8), name="cumulative area",
                showlegend=False,
                hovertemplate=("p = %{x:.3f}<br>cumulative area = %{y:.4g}"
                               "<extra></extra>"),
            ), row=area_row, col=j)
            fig.add_hline(y=0.0, line=dict(color=theme.axis, width=1),
                          row=area_row, col=j)
        fig.add_trace(case, row=q_row, col=j)
        fig.add_trace(ctrl, row=q_row, col=j)
        fig.update_yaxes(type="log", dtick=1, minor=dict(showgrid=False), row=q_row, col=j)
        # No explicit x range: p already spans [0, 1], and a fixed range here
        # breaks plotly.js's autorange on the matched y axes above.
        fig.update_xaxes(title_text="quantile p", row=q_row, col=j)
    # The dashed reference is a legend entry drawn with an empty trace, so the
    # figure explains itself without a caption.
    label = ("fitted global shift" if any(p.log2_fitted_shift is not None for p in panels)
             else "median(R): fitted global shift")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color=theme.muted, width=1, dash="dash"),
                             name=label), row=1, col=1)
    if any(p.affected_span is not None for p in panels):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=9, symbol="square", color=theme.case,
                                             opacity=0.16),
                                 name="affected fraction, 95% CI"), row=1, col=1)
    fig.update_yaxes(title_text="log₂(Q<sub>case</sub> / Q<sub>ctrl</sub>)", row=1, col=1)
    if area_row is not None:
        fig.update_yaxes(title_text="cumulative area", row=area_row, col=1)
    fig.update_yaxes(title_text="expression", row=q_row, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.yshift = 8
    _plotly_layout(fig, theme, title=title, width=width or (min(1700, 200 + 330 * k)),
                   height=height or (800 if cumulative_area else 600), legend_below=True)
    fig.update_layout(margin=dict(t=110 if title else 90))
    fig.update_layout(hovermode="x unified")
    return fig


def _marker_plotly(data, go, theme):
    """Marker spec for a point cloud: colour ramp when a colour column exists,
    a single hue otherwise; a thin white ring so overlapping points stay
    separable."""
    base = dict(size=6, line=dict(width=0.5, color=theme.surface), opacity=0.85)
    if data.color is None:
        return dict(base, color=theme.case)
    spec = data.color_spec
    return dict(base, color=data.color, colorscale=theme.scale(spec["role"]),
                cmin=spec["range"][0], cmax=spec["range"][1],
                colorbar=dict(title=dict(text=spec["label"]), thickness=12, len=0.6,
                              outlinewidth=0, tickfont=dict(color=theme.muted)))


def _hover_plotly(data):
    """A customdata block and a template listing every statistic WADE reports.

    Requested per-gene metadata leads, formatted as text rather than as a
    number — a symbol and a biotype are strings, so they cannot share the
    float64 block the statistics use.
    """
    meta = list(data.meta.items())
    cols = data.hover
    names = [k for k in cols if k != "gene"]
    blocks = [np.asarray(v, dtype=object).reshape(-1, 1) for _, v in meta]
    if names:
        blocks.append(np.column_stack([np.asarray(cols[k], dtype=np.float64) for k in names]))
    cd = np.hstack(blocks) if blocks else None
    lines = ["<b>%{text}</b>"]
    for i, (k, _) in enumerate(meta):
        lines.append(f"{k} = %{{customdata[{i}]}}")
    for i, k in enumerate(names, start=len(meta)):
        fmt = ".3g" if k.startswith("p") else ".3f"
        lines.append(f"{k} = %{{customdata[{i}]:{fmt}}}")
    return cd, "<br>".join(lines) + "<extra></extra>"


def _cloud_trace(data, go, theme, *, name="genes"):
    """One scatter trace; WebGL past a few thousand points so a transcriptome
    stays responsive."""
    n = data.gene.shape[0]
    cls = go.Scattergl if n > 2000 else go.Scatter
    cd, template = _hover_plotly(data)
    return cls(x=data.x, y=data.y, mode="markers", text=data.display, customdata=cd,
               hovertemplate=template, marker=_marker_plotly(data, go, theme), name=name,
               showlegend=False)


def _labels_plotly(data, go, fig, theme, row=None, col=None):
    idx, dx, dy = _label_positions(data)
    if idx.size == 0:
        return
    # The offsets already carry the placement, so the text is centred on its
    # resolved position rather than anchored to a side of the point.
    fig.add_trace(go.Scatter(
        x=data.x[idx] + dx, y=data.y[idx] + dy, mode="text", text=data.display[idx],
        textposition="middle center", textfont=dict(size=11, color=theme.ink),
        hoverinfo="skip", showlegend=False,
    ), row=row, col=col)


def _intervals_plotly(data, go, fig, theme, row=None, col=None):
    """Bootstrap intervals on the **labelled** points, as error bars.

    Only the labelled ones, deliberately: a transcriptome of error bars is
    mush, every gene's interval is already in the hover, and ``label=`` is
    exactly the "these are the genes I care about" the caller has already
    given. A one-pixel marker carries the bars so the coloured point beneath
    stays the point.
    """
    idx = np.flatnonzero(data.labelled)
    if idx.size == 0 or (data.x_ci is None and data.y_ci is None):
        return

    def arms(ci, v):
        if ci is None:
            return None
        return dict(type="data", symmetric=False, array=ci[1][idx] - v[idx],
                    arrayminus=v[idx] - ci[0][idx], color=theme.muted,
                    thickness=1, width=3)

    fig.add_trace(go.Scatter(
        x=data.x[idx], y=data.y[idx], mode="markers",
        marker=dict(size=1, color=theme.muted),
        error_x=arms(data.x_ci, data.x), error_y=arms(data.y_ci, data.y),
        hoverinfo="skip", showlegend=False,
    ), row=row, col=col)


def _volcano_plotly(data, theme, *, title, width, height):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = len(data)
    # rows=1: plotly's shared_xaxes links axes down COLUMNS, so it does nothing
    # here — the panels are linked explicitly with matches= below.
    fig = make_subplots(rows=1, cols=k, shared_yaxes=False,
                        subplot_titles=[f"{_STAGE_LABELS[d.stage]} stage" for d in data] if k > 1 else None,
                        horizontal_spacing=0.08)
    cloud_idx = []
    for j, d in enumerate(data, start=1):
        fig.add_trace(_cloud_trace(d, go, theme), row=1, col=j)
        cloud_idx.append(len(fig.data) - 1)
        # One-sided alternative: shade the half of the fold-change axis the test
        # cannot find anything on, so emptiness there reads as untested. Added
        # after the trace: plotly skips shapes on subplots that are still empty.
        if d.alternative != "two-sided":
            x_lim = float(np.nanmax(np.abs(d.x))) * 1.08 if np.isfinite(d.x).any() else 1.0
            x0, x1 = (0.0, x_lim) if d.alternative == "less" else (-x_lim, 0.0)
            fig.add_vrect(x0=x0, x1=x1, fillcolor=theme.grid, opacity=0.45, line_width=0,
                          layer="below", row=1, col=j,
                          annotation_text=f"untested (alternative = {d.alternative})",
                          annotation_position="bottom left" if d.alternative == "greater" else "bottom right",
                          annotation_font=dict(size=10, color=theme.muted),
                          annotation_bgcolor=theme.box())
        fig.add_vline(x=0.0, line=dict(color=theme.axis, width=1), row=1, col=j)
        if np.isfinite(d.cutoff_y):
            fig.add_hline(y=d.cutoff_y, line=dict(color=theme.muted, width=1, dash="dash"), row=1, col=j,
                          annotation_text=f"FDR {d.alpha:g}: {d.n_significant} genes",
                          annotation_position="bottom right" if d.alternative == "less" else "bottom left",
                          annotation_font=dict(size=10, color=theme.muted))
        _intervals_plotly(d, go, fig, theme, row=1, col=j)
        _labels_plotly(d, go, fig, theme, row=1, col=j)
        fig.update_xaxes(title_text=d.xlabel, row=1, col=j)
        # The promised shared x: pan/zoom one panel and the other follows, so a
        # gene's position is comparable across the two stages by eye.
        if j > 1:
            fig.update_xaxes(matches="x", row=1, col=j)
        fig.update_yaxes(title_text=d.ylabel, row=1, col=j)
    if k > 1 and data[0].color is not None:
        # One colour bar for both panels: the same column, the same scale.
        for idx in cloud_idx[:-1]:
            fig.data[idx].marker.showscale = False
    _plotly_layout(fig, theme, title=title, width=width or (520 * k + 120), height=height or 520)
    return fig


def _stages_plotly(data, theme, *, title, width, height):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(_cloud_trace(data, go, theme))
    fig.add_vline(x=data.cutoff_x, line=dict(color=theme.muted, width=1, dash="dash"))
    fig.add_hline(y=data.cutoff_y, line=dict(color=theme.muted, width=1, dash="dash"))
    # Headroom, so the upper quadrant labels do not sit on the genes they count.
    finite = data.y[np.isfinite(data.y)]
    if finite.size:
        fig.update_yaxes(range=[-0.04 * finite.max(), finite.max() * 1.22])
    counts = data.counts()
    # Corner labels: the reading table, in place. Positions in paper fractions.
    corners = {
        (True, False): dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom"),
        (True, True): dict(x=0.99, y=0.98, xanchor="right", yanchor="top"),
        (False, True): dict(x=0.01, y=0.98, xanchor="left", yanchor="top"),
        (False, False): dict(x=0.01, y=0.02, xanchor="left", yanchor="bottom"),
    }
    for key, text in QUADRANTS.items():
        pos = corners[key]
        n = counts[text.replace("\n", " ")]
        fig.add_annotation(xref="x domain", yref="y domain", showarrow=False,
                           text=f"<b>{text.replace(chr(10), '<br>')}</b><br>{n} genes",
                           font=dict(size=11, color=theme.muted), align="left" if pos["xanchor"] == "left" else "right",
                           bgcolor=theme.box(), borderpad=3, **pos)
    _labels_plotly(data, go, fig, theme)
    fig.update_xaxes(title_text=data.xlabel)
    fig.update_yaxes(title_text=data.ylabel)
    _plotly_layout(fig, theme, title=title or f"the two stages (FDR {data.alpha:g})",
                   width=width or 700, height=height or 600)
    return fig


def _drivers_plotly(panel, theme, *, title, width, height):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = panel.sample.size
    y = np.arange(k, dtype=float)
    fig = make_subplots(rows=1, cols=len(DRIVER_PANELS), shared_yaxes=True,
                        subplot_titles=[lbl for _, lbl in DRIVER_PANELS],
                        horizontal_spacing=0.035)
    for j, (field_name, _) in enumerate(DRIVER_PANELS, start=1):
        v = np.asarray(getattr(panel, field_name), dtype=np.float64)
        q25, med, q75 = panel.cohort[field_name]
        fig.add_trace(go.Scatter(
            x=v, y=y, mode="markers", text=panel.sample,
            marker=dict(size=8, color=theme.case, line=dict(width=0.5, color=theme.surface)),
            hovertemplate=("<b>%{text}</b><br>" + field_name +
                           " = %{x:.4g}<br>cohort median " + f"{med:.4g}<extra></extra>"),
            showlegend=False,
        ), row=1, col=j)
        # After the trace: plotly skips shapes on subplots that are still empty.
        fig.add_vrect(x0=q25, x1=q75, fillcolor=theme.grid, opacity=0.55, line_width=0,
                      layer="below", row=1, col=j)
        fig.add_vline(x=med, line=dict(color=theme.muted, width=1, dash="dash"),
                      row=1, col=j)
        if panel.log_axis(field_name):
            fig.update_xaxes(type="log", dtick=1, minor=dict(showgrid=False), row=1, col=j)
        fig.update_yaxes(tickvals=y, ticktext=panel.sample if j == 1 else [""] * k,
                         autorange="reversed", row=1, col=j)
    # The cohort context is the point of the figure, so it says so in the legend.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=9, symbol="square", color=theme.grid),
                             name="cohort interquartile range"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color=theme.muted, width=1, dash="dash"),
                             name="cohort median"), row=1, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 11
    _plotly_layout(fig, theme, title=title or f"{panel.gene}: {panel.subtitle}",
                   width=width or 1080, height=height or max(320, 130 + 26 * k),
                   legend_below=True)
    fig.update_layout(margin=dict(l=140, r=30, t=90, b=70))
    return fig


def _linked_plotly(data, panel_of, theme, *, title, width, height):
    """A volcano wired to a gene panel: click a point, the panel redraws.

    ``panel_of(i)`` builds the :class:`~wade.plotting.GenePanel` for gene index
    ``i``; the click handler closes over it rather than over a result, which
    keeps this module ignorant of the core. The panel's traces are rewritten in
    place inside one ``batch_update`` — a redraw, not a rebuild — so the
    volcano's zoom and the reader's place are not lost on every click.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    try:
        widget = go.FigureWidget(make_subplots(
            rows=2, cols=2, specs=[[{"rowspan": 2}, {}], [None, {}]],
            column_widths=[0.56, 0.44], row_heights=[0.55, 0.45],
            subplot_titles=[f"{_STAGE_LABELS[data.stage]} stage", "click a point", ""],
            horizontal_spacing=0.11, vertical_spacing=0.12))
    except ImportError as e:
        raise ImportError(
            f"a linked view needs plotly's FigureWidget, which needs anywidget: {e}. "
            f"Install it with `conda install anywidget` or `pip install "
            f"'wade[linked]'`. Every static figure works without it."
        ) from e

    widget.add_trace(_cloud_trace(data, go, theme), row=1, col=1)
    if np.isfinite(data.cutoff_y):
        widget.add_hline(y=data.cutoff_y, line=dict(color=theme.muted, width=1, dash="dash"),
                         row=1, col=1)
    widget.add_vline(x=0.0, line=dict(color=theme.axis, width=1), row=1, col=1)
    _labels_plotly(data, go, widget, theme, row=1, col=1)
    widget.update_xaxes(title_text=data.xlabel, row=1, col=1)
    widget.update_yaxes(title_text=data.ylabel, row=1, col=1)

    # The panel side, seeded on the first gene so the traces exist to update.
    # The fitted-shift reference is a trace, not a shape: it moves per gene, and
    # a trace is what batch_update can rewrite.
    ratio, case, ctrl = _gene_traces(panel_of(0), theme, go, showlegend=False)
    widget.add_trace(ratio, row=1, col=2)
    widget.add_trace(go.Scatter(x=[0.0, 1.0], y=[0.0, 0.0], mode="lines",
                                line=dict(color=theme.muted, width=1, dash="dash"),
                                hoverinfo="skip", showlegend=False), row=1, col=2)
    widget.add_hline(y=0.0, line=dict(color=theme.axis, width=1), row=1, col=2)
    widget.add_trace(case, row=2, col=2)
    widget.add_trace(ctrl, row=2, col=2)
    widget.update_yaxes(title_text="log₂(Q<sub>case</sub> / Q<sub>ctrl</sub>)", row=1, col=2)
    widget.update_yaxes(type="log", dtick=1, minor=dict(showgrid=False),
                        title_text="expression", row=2, col=2)
    widget.update_xaxes(title_text="quantile p", row=2, col=2)
    curve, reference, case_t, ctrl_t = widget.data[-4:]

    def redraw(_trace, points, _state):
        if not points.point_inds:
            return
        p = panel_of(int(points.point_inds[0]))
        d = p.detail
        with widget.batch_update():
            curve.x, curve.y = d.p, d.r
            curve.customdata = np.c_[d.y1, d.y0]
            reference.y = (p.reference, p.reference)
            case_t.x, case_t.y = d.p, d.y1
            ctrl_t.x, ctrl_t.y = d.p, d.y0
            # One annotation, two lines, exactly as _gene_plotly titles a panel.
            # An empty subplot title is dropped by make_subplots, so there is no
            # second slot to put the subtitle in.
            widget.layout.annotations[1].text = (
                f"<b>{p.name}</b><br><span style='font-size:10px;"
                f"color:{theme.muted}'>{p.subtitle}</span>")

    widget.data[0].on_click(redraw)
    for ann in widget.layout.annotations:
        ann.font.size = 12
        ann.yshift = 8
    _plotly_layout(widget, theme, title=title or "click a gene to see its panel",
                   width=width or 1180, height=height or 620)
    widget.update_layout(margin=dict(t=100))
    return widget
