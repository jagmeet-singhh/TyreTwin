import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List


def create_degradation_chart(pred_data: Dict[str, Any]) -> go.Figure:
    """
    Renders the 4-Corner Gaussian Process degradation pace curves:
    - FL (Front-Left), FR (Front-Right), RL (Rear-Left), RR (Rear-Right)
    Displays distinct ML-predicted pace, confidence envelopes, and clean lap observations.
    """
    fig = go.Figure()
    
    curve = pred_data.get("curve", [])
    obs = pred_data.get("observed_laps", [])
    clean = pred_data.get("clean_laps", [])
    compound = pred_data.get("compound", "MEDIUM")
    four_tyres = pred_data.get("four_tyres", {})
    corner_curves = four_tyres.get("corner_gp_curves", {})
    limiting = four_tyres.get("limiting_corner", "RL")

    corner_configs = {
        "FL": {"name": "FL Pace (Front-Left)", "color": "#00d2be", "dash": "solid"},
        "FR": {"name": "FR Pace (Front-Right)", "color": "#3498db", "dash": "solid"},
        "RL": {"name": "RL Pace (Rear-Left)", "color": "#ffd700", "dash": "solid"},
        "RR": {"name": "RR Pace (Rear-Right)", "color": "#ff793f", "dash": "solid"},
    }

    # 1. Plot 4 distinct Corner GP Curves if available
    if corner_curves:
        # Plot limiting corner confidence band first
        if limiting in corner_curves and corner_curves[limiting]:
            df_lim = pd.DataFrame(corner_curves[limiting])
            fig.add_trace(go.Scatter(
                x=list(df_lim["tyre_age"]) + list(df_lim["tyre_age"])[::-1],
                y=list(df_lim["upper_bound_sec"]) + list(df_lim["lower_bound_sec"])[::-1],
                fill="toself",
                fillcolor="rgba(255, 24, 1, 0.10)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name=f"{limiting} (Limiter) ±2σ Confidence Band"
            ))

        for corner, cfg in corner_configs.items():
            if corner in corner_curves and corner_curves[corner]:
                df_c = pd.DataFrame(corner_curves[corner])
                is_lim = (corner == limiting)
                width = 3.6 if is_lim else 2.2
                name_lbl = f"{cfg['name']} [LIMITER]" if is_lim else cfg["name"]

                fig.add_trace(go.Scatter(
                    x=df_c["tyre_age"],
                    y=df_c["predicted_lap_time_sec"],
                    mode="lines",
                    name=name_lbl,
                    line=dict(color=cfg["color"], width=width, dash=cfg["dash"]),
                    hovertemplate=f"<b>{corner}</b> (Lap %{{x}})<br>Pace: %{{y:.3f}}s<extra></extra>"
                ))
    elif curve:
        # Fallback to single GP curve if corners not provided
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
        fig.add_trace(go.Scatter(
            x=df_curve["tyre_age"],
            y=df_curve["predicted_lap_time_sec"],
            mode="lines",
            name=f"Predicted {compound} Degradation",
            line=dict(color="#ffd700", width=3.5),
            hovertemplate="Tyre Age: %{x} Laps<br>Predicted Pace: %{y:.3f}s<extra></extra>"
        ))

    # 2. Clean Filtered Lap Times (Noise Removed)
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

    # 3. Raw Observed Lap Times
    if obs:
        df_obs = pd.DataFrame(obs)
        fig.add_trace(go.Scatter(
            x=df_obs["tyre_age_lap"],
            y=df_obs["lap_time_sec"],
            mode="markers",
            name="Raw Observed Laps (Noisy)",
            marker=dict(color="rgba(255, 255, 255, 0.35)", size=6, symbol="circle-open"),
            hovertemplate="Observed: %{y:.3f}s (+%{text:.2f}s Noise)<extra></extra>",
            text=df_obs.get("predicted_noise_sec", [0.0]*len(df_obs))
        ))

    # Pit Window & Cliff Annotations
    cliff_lap = pred_data.get("cliff_lap", 26)
    rec_pit = pred_data.get("recommended_pit_lap", 24)
    
    fig.add_vline(x=rec_pit, line_width=1.5, line_dash="dash", line_color="#00d2be", annotation_text="Optimal Pit", annotation_position="top left")
    fig.add_vline(x=cliff_lap, line_width=1.5, line_dash="dot", line_color="#ff1801", annotation_text=f"{limiting} Tyre Cliff", annotation_position="top right")

    fig.update_layout(
        title=dict(
            text="4-CORNER (FL, FR, RL, RR) TYRE PACE DEGRADATION & CONFIDENCE",
            font=dict(family="Orbitron", size=14, color="#ffffff")
        ),
        xaxis=dict(title="Tyre Age (Laps Completed)", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(title="Lap Time (Seconds)", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        paper_bgcolor="rgba(11, 14, 20, 0.95)",
        plot_bgcolor="rgba(11, 14, 20, 0.95)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#8c9ba5", size=11)),
        hovermode="x unified",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
