import plotly.graph_objects as go
from typing import List, Dict, Any


def create_shap_waterfall_chart(attributions: List[Dict[str, Any]], base_pace: float = 91.20) -> go.Figure:
    names = [a.get("display_name", a.get("feature", "")) for a in attributions]
    deltas = [a.get("delta_sec", 0.0) for a in attributions]
    
    colors = ["#e10600" if d > 0 else "#00d2be" for d in deltas]

    fig = go.Figure(go.Bar(
        x=deltas,
        y=names,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{d:+.3f}s" for d in deltas],
        textposition="auto",
        hovertemplate="%{y}: %{x:+.3f}s delta<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="SHAP EXPLAINABILITY — EXTERNAL NOISE DECOUPLING", font=dict(family="Orbitron", size=14, color="#ffffff")),
        xaxis=dict(title="Lap Time Delta Impact (Seconds)", gridcolor="rgba(255,255,255,0.06)", zerolinecolor="#ffffff"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.06)"),
        paper_bgcolor="rgba(11, 14, 20, 0.95)",
        plot_bgcolor="rgba(11, 14, 20, 0.95)",
        font=dict(color="#8c9ba5", family="Titillium Web"),
        height=320,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
