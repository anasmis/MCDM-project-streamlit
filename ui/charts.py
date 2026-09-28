import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#22634B", "#B38949", "#6A8792", "#8C6B77", "#8B9661", "#555F76"]


def polish(figure: go.Figure, height: int = 320) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        font={"family": "Arial, sans-serif", "size": 12, "color": "#52604f"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 15, "b": 30, "l": 0, "r": 30},
        xaxis={"gridcolor": "#e5e7df", "zeroline": False},
        yaxis={"gridcolor": "#e5e7df", "zeroline": False},
        legend={"title": None, "orientation": "h", "y": -0.2},
        separators=", ",
    )
    return figure


def weight_chart(names, weights) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=weights,
            y=names,
            orientation="h",
            marker_color=PALETTE[0],
            text=[f"{w:.1%}".replace(".", ",") for w in weights],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y} : %{x:.1%}<extra></extra>",
        )
    )
    polish(figure, max(210, len(names) * 35 + 65))
    figure.update_xaxes(range=[0, max(weights) * 1.2], tickformat=".0%", title=None)
    figure.update_yaxes(autorange="reversed", title=None)
    return figure


def ranking_chart(frame) -> go.Figure:
    colors = [PALETTE[0] if rank == 1 else "#afbdac" for rank in frame["Rang"]]
    figure = go.Figure(
        go.Bar(
            x=frame["Score"],
            y=frame["Alternative"],
            orientation="h",
            marker_color=colors,
            text=[f"{s:.3f}".replace(".", ",") for s in frame["Score"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y} : %{x:.4f}<extra></extra>",
        )
    )
    polish(figure, max(250, len(frame) * 38 + 65))
    figure.update_xaxes(range=[0, 1.12], title="Score · plus élevé = mieux classé")
    figure.update_yaxes(autorange="reversed", title=None)
    return figure


def sensitivity_chart(frame) -> go.Figure:
    figure = px.line(
        frame,
        x="Variation (%)",
        y="Rang",
        color="Alternative",
        markers=True,
        color_discrete_sequence=PALETTE,
        hover_data={"Score": ":.4f", "Poids du critère": ":.1%"},
    )
    polish(figure, 360)
    figure.update_yaxes(autorange="reversed", dtick=1)
    figure.update_xaxes(ticksuffix=" %")
    return figure
