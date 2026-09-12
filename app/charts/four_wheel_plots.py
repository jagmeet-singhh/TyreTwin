import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, Optional


def create_four_wheel_degradation_chart(four_tyres_data: Optional[Dict[str, Any]]) -> go.Figure:
    """Plots 4-corner remaining rubber % degradation curves over stint laps."""
    fig = go.Figure()
    if not four_tyres_data or "curves" not in four_tyres_data:
        fig.update_layout(title="No 4-Wheel Degradation Telemetry Available")
        return fig

    curves = four_tyres_data.get("curves", {})
    limiting = four_tyres_data.get("limiting_corner", "RL")
    tyres = four_tyres_data.get("tyres", {})

    corner_configs = {
        "FL": {"name": "Front-Left (FL)", "color": "#00d2be", "dash": "solid"},
        "FR": {"name": "Front-Right (FR)", "color": "#3498db", "dash": "solid"},
        "RL": {"name": "Rear-Left (RL)", "color": "#ffd700", "dash": "solid"},
        "RR": {"name": "Rear-Right (RR)", "color": "#ff793f", "dash": "solid"},
    }

    for corner, cfg in corner_configs.items():
        if corner in curves and curves[corner]:
            df = pd.DataFrame(curves[corner])
            is_lim = (corner == limiting)
            width = 3.8 if is_lim else 2.2
            name_label = f"{cfg['name']} ★ LIMITER" if is_lim else cfg["name"]

            fig.add_trace(
                go.Scatter(
                    x=df["lap"],
                    y=df["remaining_wear_pct"],
                    mode="lines",
                    name=name_label,
                    line=dict(color=cfg["color"], width=width, dash=cfg["dash"]),
                    hovertemplate=f"<b>{corner}</b> (Lap %{{x}})<br>Remaining Life: %{{y:.1f}}%<extra></extra>",
                )
            )

    # 20% Critical Cliff Threshold
    fig.add_hline(
        y=20.0,
        line_dash="dot",
        line_color="#ff1801",
        line_width=1.8,
        annotation_text="Critical Tyre Cliff (20% Rubber Remaining)",
        annotation_position="bottom right",
        annotation_font=dict(color="#ff1801", size=11),
    )

    # Limiting Tyre Cliff Lap
    lim_cliff = tyres.get(limiting, {}).get("cliff_lap", 24)
    fig.add_vline(
        x=lim_cliff,
        line_dash="dash",
        line_color="#ff1801",
        line_width=1.5,
        annotation_text=f"{limiting} Cliff: Lap {lim_cliff}",
        annotation_position="top left",
        annotation_font=dict(color="#ff1801", size=11),
    )

    fig.update_layout(
        title=dict(
            text="4-CORNER ASYMMETRIC TYRE WEAR DEGRADATION (FL, FR, RL, RR)",
            font=dict(family="Orbitron", size=14, color="#ffffff"),
        ),
        xaxis=dict(title="Stint Lap Number", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(
            title="Remaining Tyre Rubber Health (%)",
            range=[0, 105],
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        paper_bgcolor="rgba(11, 14, 20, 0.95)",
        plot_bgcolor="rgba(11, 14, 20, 0.95)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#8c9ba5", size=11),
        ),
        hovermode="x unified",
        height=420,
        margin=dict(l=40, r=40, t=55, b=40),
    )
    return fig


def create_four_wheel_thermal_chart(four_tyres_data: Optional[Dict[str, Any]]) -> go.Figure:
    """Plots 4-corner surface temperatures and thermodynamic operating windows."""
    fig = go.Figure()
    if not four_tyres_data or "curves" not in four_tyres_data:
        fig.update_layout(title="No Thermal Telemetry Available")
        return fig

    curves = four_tyres_data.get("curves", {})
    limiting = four_tyres_data.get("limiting_corner", "RL")
    tyres = four_tyres_data.get("tyres", {})
    sample_tyre = tyres.get(limiting, tyres.get("FL", {}))
    t_min = sample_tyre.get("temp_opt_min", 105.0)
    t_max = sample_tyre.get("temp_opt_max", 125.0)

    # Shaded Optimal Thermal Window
    max_laps = len(curves.get("FL", []))
    if max_laps > 0:
        fig.add_hrect(
            y0=t_min,
            y1=t_max,
            fillcolor="rgba(0, 210, 190, 0.12)",
            line_width=0,
            annotation_text=f"Optimal Working Window ({t_min:.0f}°C - {t_max:.0f}°C)",
            annotation_position="top left",
            annotation_font=dict(color="#00d2be", size=10),
        )

    corner_colors = {"FL": "#00d2be", "FR": "#3498db", "RL": "#ffd700", "RR": "#ff793f"}

    for corner, col in corner_colors.items():
        if corner in curves and curves[corner]:
            df = pd.DataFrame(curves[corner])
            fig.add_trace(
                go.Scatter(
                    x=df["lap"],
                    y=df["surface_temp_c"],
                    mode="lines",
                    name=f"{corner} Surface Temp",
                    line=dict(color=col, width=2.4),
                    hovertemplate=f"<b>{corner}</b>: %{{y:.1f}}°C (Lap %{{x}})<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(
            text="FOUR-CORNER SURFACE TEMPERATURE TRAJECTORY (°C)",
            font=dict(family="Orbitron", size=14, color="#ffffff"),
        ),
        xaxis=dict(title="Stint Lap Number", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(
            title="Surface Temperature (°C)",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        paper_bgcolor="rgba(11, 14, 20, 0.95)",
        plot_bgcolor="rgba(11, 14, 20, 0.95)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#8c9ba5", size=11),
        ),
        hovermode="x unified",
        height=400,
        margin=dict(l=40, r=40, t=55, b=40),
    )
    return fig
