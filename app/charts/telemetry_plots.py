import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List


def create_telemetry_multitrace_chart(tel_data: Dict[str, Any]) -> go.Figure:
    trace = tel_data.get("trace", [])
    if not trace:
        fig = go.Figure()
        fig.update_layout(title="No Telemetry Available for Selected Lap")
        return fig

    df = pd.DataFrame(trace)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.35, 0.25, 0.20, 0.20],
        subplot_titles=("Speed (km/h)", "Throttle (%) & Brake (%)", "Gear & DRS", "Engine RPM")
    )

    # Row 1: Speed
    fig.add_trace(
        go.Scatter(x=df["distance_m"], y=df["speed_kmh"], name="Speed", line=dict(color="#00d2be", width=2)),
        row=1, col=1
    )

    # Row 2: Throttle & Brake
    fig.add_trace(
        go.Scatter(x=df["distance_m"], y=df["throttle_pct"], name="Throttle", line=dict(color="#39b54a", width=1.8)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["distance_m"], y=df["brake_pct"], name="Brake", line=dict(color="#e10600", width=1.8)),
        row=2, col=1
    )

    # Row 3: Gear & DRS
    fig.add_trace(
        go.Scatter(x=df["distance_m"], y=df["gear"], name="Gear", line=dict(color="#ffd700", width=1.8)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["distance_m"], y=df["drs"] * 8, name="DRS Active", line=dict(color="#64c4ff", dash="dash")),
        row=3, col=1
    )

    # Row 4: Engine RPM
    fig.add_trace(
        go.Scatter(x=df["distance_m"], y=df["rpm"], name="RPM", line=dict(color="#a0aec0", width=1.5)),
        row=4, col=1
    )

    fig.update_layout(
        paper_bgcolor="rgba(11, 14, 20, 0.95)",
        plot_bgcolor="rgba(11, 14, 20, 0.95)",
        font=dict(color="#8c9ba5", family="Titillium Web"),
        height=620,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        showlegend=False
    )
    fig.update_xaxes(title_text="Track Distance (Meters)", row=4, col=1, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig
