"""The plotly renderer: interactive figures, one function per public figure.

``import plotly`` happens inside each renderer, not here, so that importing
this module costs nothing and ``import wade`` stays NumPy-only. ``go`` is
passed down to the trace helpers for the same reason.
"""

from __future__ import annotations

import numpy as np

from .data import _label_positions
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


def _gene_plotly(panels, theme, *, share_y, title, width, height):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = len(panels)
    titles = []
    for p in panels:
        sub = "<br>".join(p.subtitle_lines)
        titles.append(f"<b>{p.name}</b><br><span style='font-size:10px;color:{theme.muted}'>{sub}</span>")
    fig = make_subplots(rows=2, cols=k, shared_xaxes=True, shared_yaxes=False,
                        subplot_titles=titles, row_heights=[0.55, 0.45],
                        vertical_spacing=0.12, horizontal_spacing=0.06 if k > 1 else 0.05)
    if share_y:
        # Only the log-ratio row shares a scale: magnitudes of R are comparable
        # across genes, raw expression levels are not.
        for j in range(2, k + 1):
            fig.update_yaxes(matches="y", row=1, col=j)
    for j, p in enumerate(panels, start=1):
        d = p.detail
        fig.add_trace(go.Scatter(
            x=d.p, y=d.r, mode="lines", line=dict(color=theme.ink, width=1.8),
            name="log₂ ratio", legendgroup="r", showlegend=(j == 1),
            customdata=np.c_[d.y1, d.y0],
            hovertemplate=("p = %{x:.3f}<br>log₂ ratio = %{y:.2f}<br>"
                           "Q<sub>case</sub> = %{customdata[0]:.3g}<br>Q<sub>ctrl</sub> = %{customdata[1]:.3g}"
                           "<extra></extra>"),
        ), row=1, col=j)
        fig.add_hline(y=0.0, line=dict(color=theme.axis, width=1), row=1, col=j)
        fig.add_hline(y=p.reference, line=dict(color=theme.muted, width=1, dash="dash"), row=1, col=j)
        # How firm the extent of the affected region is. Added after the trace:
        # plotly skips shapes on subplots that are still empty.
        span = p.affected_span
        if span is not None:
            fig.add_vrect(x0=span[0], x1=span[1], fillcolor=theme.case, opacity=0.16,
                          line_width=0, layer="below", row=1, col=j)
        fig.add_trace(go.Scatter(
            x=d.p, y=d.y1, mode="lines", line=dict(color=theme.case, width=1.8),
            name="case", legendgroup="case", showlegend=(j == 1),
            hovertemplate="p = %{x:.3f}<br>case Q = %{y:.3g}<extra></extra>",
        ), row=2, col=j)
        fig.add_trace(go.Scatter(
            x=d.p, y=d.y0, mode="lines", line=dict(color=theme.ctrl, width=1.8),
            name="control", legendgroup="ctrl", showlegend=(j == 1),
            hovertemplate="p = %{x:.3f}<br>control Q = %{y:.3g}<extra></extra>",
        ), row=2, col=j)
        fig.update_yaxes(type="log", dtick=1, minor=dict(showgrid=False), row=2, col=j)
        # No explicit x range: p already spans [0, 1], and a fixed range here
        # breaks plotly.js's autorange on the matched y axes above.
        fig.update_xaxes(title_text="quantile p", row=2, col=j)
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
    fig.update_yaxes(title_text="expression", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.yshift = 8
    _plotly_layout(fig, theme, title=title, width=width or (min(1700, 200 + 330 * k)),
                   height=height or 600, legend_below=True)
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
    """A customdata block and a template listing every statistic WADE reports."""
    cols = data.hover
    names = [k for k in cols if k != "gene"]
    cd = np.column_stack([np.asarray(cols[k], dtype=np.float64) for k in names]) if names else None
    lines = ["<b>%{text}</b>"]
    for i, k in enumerate(names):
        fmt = ".3g" if k.startswith("p") else ".3f"
        lines.append(f"{k} = %{{customdata[{i}]:{fmt}}}")
    return cd, "<br>".join(lines) + "<extra></extra>"


def _cloud_trace(data, go, theme, *, name="genes"):
    """One scatter trace; WebGL past a few thousand points so a transcriptome
    stays responsive."""
    n = data.gene.shape[0]
    cls = go.Scattergl if n > 2000 else go.Scatter
    cd, template = _hover_plotly(data)
    return cls(x=data.x, y=data.y, mode="markers", text=data.gene, customdata=cd,
               hovertemplate=template, marker=_marker_plotly(data, go, theme), name=name,
               showlegend=False)


def _labels_plotly(data, go, fig, theme, row=None, col=None):
    idx, above = _label_positions(data)
    if idx.size == 0:
        return
    for sel, pos in ((above, "top center"), (~above, "bottom center")):
        if not sel.any():
            continue
        fig.add_trace(go.Scatter(
            x=data.x[idx[sel]], y=data.y[idx[sel]], mode="text", text=data.gene[idx[sel]],
            textposition=pos, textfont=dict(size=11, color=theme.ink),
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
