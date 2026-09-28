"""Interactive versions of the same empirical curves and effect plots."""
import numpy as np
from .data import EXPRESSION_LABEL


def _style(fig, spec):
    t = spec["theme"]
    fig.update_layout(template=t.plotly_template, title=spec["title"],
                      paper_bgcolor=t.surface, plot_bgcolor=t.surface,
                      font=dict(family=t.font, color=t.ink), height=440,
                      margin=dict(l=65, r=25, t=75, b=65))
    fig.update_xaxes(gridcolor=t.grid, linecolor=t.axis)
    fig.update_yaxes(gridcolor=t.grid, linecolor=t.axis)
    return fig


def gene(spec):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = spec["theme"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.13)
    for value, name, color in [(1, "Case", t.case), (0, "Control", t.ctrl)]:
        x = np.sort(spec["expression"][spec["cond"] == value])
        fig.add_trace(go.Scatter(x=np.r_[x[0],x], y=np.r_[0,np.arange(1,len(x)+1)/len(x)],
                                mode="lines", line=dict(color=color, shape="hv"), name=f"{name} (n={len(x)})"),row=1,col=1)
    d = spec["detail"]
    fig.add_trace(go.Scatter(x=d.probability, y=np.r_[d.effect,d.effect[-1]], mode="lines",
                            line=dict(color=t.ink, shape="hv"), name="Quantile effect"),row=1,col=2)
    for edge, label, style in spec["boundaries"]:
        fig.add_vline(x=edge, line_dash="dash" if style == "--" else "dot", line_color=t.ink, row=1, col=1,
                      annotation_text=label, annotation_position="top left")
    fig.add_hline(y=spec["bulk"], line_dash="dash", line_color=t.case, row=1, col=2)
    fig.update_xaxes(title_text=EXPRESSION_LABEL,row=1,col=1)
    fig.update_yaxes(title_text="Cumulative fraction",row=1,col=1)
    fig.update_xaxes(title_text="Expression quantile",row=1,col=2)
    fig.update_yaxes(title_text="Case − control (log₂ units)",row=1,col=2)
    return _style(fig, spec)


def scatter(spec):
    import plotly.graph_objects as go
    t = spec["theme"]
    fig = go.Figure()
    for mask, color, name in [(spec["keep"] & ~spec["significant"], t.muted, "Not called"),
                              (spec["significant"], t.case, spec["called"])]:
        fig.add_trace(go.Scattergl(x=spec["x"][mask], y=spec["y"][mask], text=spec["gene"][mask], mode="markers",
                                  marker=dict(color=color,size=6,opacity=.65),name=name,
                                  hovertemplate="%{text}<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>"))
    for i in spec["labelled"]:
        if spec["keep"][i]:
            fig.add_annotation(x=spec["x"][i],y=spec["y"][i],text=str(spec["gene"][i]),showarrow=False,yshift=10)
    if spec["kind"] == "volcano":
        fig.add_vline(x=0,line_color=t.axis)
    fig.add_hline(y=-np.log10(spec["alpha"]) if spec["kind"] == "volcano" else 0,line_color=t.axis,line_dash="dash")
    fig.update_xaxes(title_text=spec["xlabel"])
    fig.update_yaxes(title_text=spec["ylabel"])
    return _style(fig, spec)
