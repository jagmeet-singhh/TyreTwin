import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List


def create_degradation_chart(pred_data: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    
    curve = pred_data.get("curve", [])
    obs = pred_data.get("observed_laps", [])
    clean = pred_data.get("clean_laps", [])
    compound = pred_data.get("compound", "MEDIUM")

    comp_colors = {
        "SOFT": "#ff1801",
        "MEDIUM": "#ffd700",
        "HARD": "#ffffff",
        "INTERMEDIATE": "#39b54a",
        "WET": "#00a0de"
    }
    line_color = comp_colors.get(compound.upper(), "#ffd700")

    # 1. Uncertainty Band (+/- 2 Sigma)
    if curve:
        df_curve = pd.DataFrame(curve)
        fig.add_trace(go.Scatter(
            x=list(df_curve["tyre_age"]) + list(df_curve["tyre_age"])[::-1],
            y=list(df_curve["upper_bound_sec"]) + list(df_curve["lower_bound_sec"])[::-1],
            fill="toself",
            fillcolor="rgba(225, 6, 0, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="±2σ Confidence Band"
        ))

        # 2. Predicted Degradation Mean Curve
        fig.add_trace(go.Scatter(
            x=df_curve["tyre_age"],
            y=df_curve["predicted_lap_time_sec"],
            mode="lines",
            name=f"Predicted {compound} Degradation",
            line=dict(color=line_color, width=3.5),
            hovertemplate="Tyre Age: %{x} Laps<br>Predicted Pace: %{y:.3f}s<extra></extra>"
        ))

    # 3. Clean Filtered Lap Times (Noise Removed)
    if clean:
        df_clean = pd.DataFrame(clean)
        fig.add_trace(go.Scatter(
            x=df_clean["tyre_age_lap"],
            y=df_clean["clean_lap_time_sec"],
            mode="markers",
            name="Clean Lap Times (Noise Decoupled)",
            marker=dict(color="#00d2be", size=8, symbol="diamond"),
            hovertemplate="Clean Lap: %{y:.3f}s (Age: %{x})<extra></extra>"
        ))

    # 4. Raw Observed Lap Times
    if obs:
        df_obs = pd.DataFrame(obs)
        fig.add_trace(go.Scatter(
            x=df_obs["tyre_age_lap"],
            y=df_obs["lap_time_sec"],
            mode="markers",
            name="Raw Observed Laps (Noisy)",
            marker=dict(color="rgba(255, 255, 255, 0.4)", size=6, symbol="circle-open"),
            hovertemplate="Observed: %{y:.3f}s (+%{text:.2f}s Noise)<extra></extra>",
            text=df_obs["predicted_noise_sec"]
        ))

    # Pit Window & Cliff Annotations
    cliff_lap = pred_data.get("cliff_lap", 26)
    rec_pit = pred_data.get("recommended_pit_lap", 24)
    
    fig.add_vline(x=rec_pit, line_width=1.5, line_dash="dash", line_color="#00d2be", annotation_text="Optimal Pit", annotation_position="top left")
    fig.add_vline(x=cliff_lap, line_width=1.5, line_dash="dot", line_color="#ff1801", annotation_text="Tyre Cliff", annotation_position="top right")

    fig.update_layout(
        title=dict(text="TYRE DEGRADATION INTELLIGENCE — TRUE PACE & CONFIDENCE", font=dict(family="Orbitron", size=15, color="#ffffff")),
        xaxis=dict(title="Tyre Age (Laps Completed)", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(title="Lap Time (Seconds)", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        paper_bgcolor="rgba(11, 14, 20, 0.95)",
        plot_bgcolor="rgba(11, 14, 20, 0.95)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#8c9ba5")),
        hovermode="x unified",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
