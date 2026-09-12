"""
TyreTwin Live Noise Removal & Telemetry Decoupling Visualizer Plots
Provides:
1. create_noise_removal_stage_chart: 5-Stage interactive step-by-step decoupling visualizer.
2. create_side_by_side_comparison_chart: Side-by-side comparison matching the reference image.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any


COMPOUND_COLORS = {
    "SOFT": "#ff1801",
    "MEDIUM": "#ffd700",
    "HARD": "#ffffff",
    "INTERMEDIATE": "#39b54a",
    "WET": "#00a0de"
}


def create_noise_removal_stage_chart(pred_data: Dict[str, Any], current_stage: int = 1) -> go.Figure:
    """
    Renders an interactive step-by-step chart illustrating the 5 stages of telemetry decoupling:
    1: Raw practice laps
    2: Traffic laps disappear
    3: Yellow flag laps disappear
    4: Fuel correction
    5: Clean degradation curve appears
    """
    fig = go.Figure()
    
    stages = pred_data.get("noise_removal_stages", {})
    compound = pred_data.get("compound", "MEDIUM").upper()
    comp_color = COMPOUND_COLORS.get(compound, "#ffd700")
    curve = pred_data.get("curve", [])
    df_curve = pd.DataFrame(curve) if curve else pd.DataFrame()
    naive_reg = stages.get("naive_regression", {"slope": 0.15, "intercept": 90.0})

    stage_1 = stages.get("stage_1_raw", [])
    stage_2 = stages.get("stage_2_traffic_filtered", [])
    stage_3 = stages.get("stage_3_incident_filtered", [])
    stage_4 = stages.get("stage_4_fuel_corrected", [])
    stage_5 = stages.get("stage_5_clean", [])

    # Titles and subtitle per stage
    stage_meta = {
        1: {
            "title": "STAGE 1/5: RAW PRACTICE LAPS (WITH NOISE & OUTLIERS)",
            "subtitle": "Raw telemetry recorded during session — contains traffic dirty air, yellow flags, in/out laps, and fuel weight decay.",
        },
        2: {
            "title": "STAGE 2/5: TRAFFIC LAPS DISAPPEAR",
            "subtitle": "Laps hindered by dirty air or traffic (+0.65s to +1.40s) are detected and filtered out.",
        },
        3: {
            "title": "STAGE 3/5: YELLOW FLAG & INCIDENT LAPS DISAPPEAR",
            "subtitle": "Slow caution periods (VSC, yellow flags) and pit in/out box laps are filtered out.",
        },
        4: {
            "title": "STAGE 4/5: FUEL MASS BURNOFF CORRECTION",
            "subtitle": "Normalized for fuel burn mass effect (-0.30s per 10kg) to dry-weight reference pace.",
        },
        5: {
            "title": "STAGE 5/5: CLEAN DEGRADATION CURVE APPEARS",
            "subtitle": "True Gaussian Process tyre degradation curve isolated with ±2σ confidence band and cliff point.",
        }
    }

    meta = stage_meta.get(current_stage, stage_meta[1])

    # STAGE 1: RAW PRACTICE LAPS
    if current_stage == 1:
        if stage_1:
            df = pd.DataFrame(stage_1)
            # Normal laps
            fig.add_trace(go.Scatter(
                x=df["tyre_age_lap"],
                y=df["lap_time_sec"],
                mode="markers",
                name="Raw Practice Laps",
                marker=dict(size=9, color="#ff4444", symbol="star", line=dict(width=1, color="#ffffff")),
                hovertemplate="Lap %{text}: %{y:.3f}s (Age: %{x})<extra>Raw Lap</extra>",
                text=df["lap_number"]
            ))
            # Naive linear regression fit showing distorted slope
            min_x, max_x = df["tyre_age_lap"].min(), df["tyre_age_lap"].max()
            x_line = np.linspace(min_x, max_x, 20)
            y_line = naive_reg["intercept"] + naive_reg["slope"] * x_line
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Naive Distorted Trend (With Outliers)",
                line=dict(color="#3b82f6", width=2.5, dash="dash"),
                hovertemplate="Naive Trend: %{y:.3f}s<extra></extra>"
            ))

    # STAGE 2: TRAFFIC LAPS DISAPPEAR
    elif current_stage == 2:
        df_all = pd.DataFrame(stage_1) if stage_1 else pd.DataFrame()
        if not df_all.empty:
            traffic_laps = df_all[df_all["traffic_flag"]]
            clean_traffic = df_all[~df_all["traffic_flag"]]
            # Remaining clean of traffic
            fig.add_trace(go.Scatter(
                x=clean_traffic["tyre_age_lap"],
                y=clean_traffic["lap_time_sec"],
                mode="markers",
                name="Free Air Practice Laps",
                marker=dict(size=9, color="#ff8800", symbol="star"),
                hovertemplate="Lap %{text}: %{y:.3f}s (Age: %{x})<extra>Free Air</extra>",
                text=clean_traffic["lap_number"]
            ))
            # Faded traffic laps marked with 'X'
            if not traffic_laps.empty:
                fig.add_trace(go.Scatter(
                    x=traffic_laps["tyre_age_lap"],
                    y=traffic_laps["lap_time_sec"],
                    mode="markers+text",
                    name="Filtered Traffic Laps",
                    marker=dict(size=12, color="rgba(255, 68, 68, 0.4)", symbol="x"),
                    text=["Traffic" for _ in range(len(traffic_laps))],
                    textposition="top center",
                    textfont=dict(size=10, color="rgba(255, 100, 100, 0.7)"),
                    hovertemplate="Filtered Lap %{customdata}: +Traffic Noise (+%{marker.size})<extra>Traffic</extra>",
                    customdata=traffic_laps["lap_number"]
                ))

    # STAGE 3: YELLOW FLAG LAPS DISAPPEAR
    elif current_stage == 3:
        df_s2 = pd.DataFrame(stage_2) if stage_2 else pd.DataFrame()
        if not df_s2.empty:
            incident_laps = df_s2[df_s2["is_incident"]]
            valid_laps = df_s2[~df_s2["is_incident"]]
            # Retained valid laps
            fig.add_trace(go.Scatter(
                x=valid_laps["tyre_age_lap"],
                y=valid_laps["lap_time_sec"],
                mode="markers",
                name="Green Flag Representative Laps",
                marker=dict(size=9, color="#ffd700", symbol="diamond"),
                hovertemplate="Lap %{text}: %{y:.3f}s (Age: %{x})<extra>Green Flag</extra>",
                text=valid_laps["lap_number"]
            ))
            # Faded incident laps
            if not incident_laps.empty:
                fig.add_trace(go.Scatter(
                    x=incident_laps["tyre_age_lap"],
                    y=incident_laps["lap_time_sec"],
                    mode="markers+text",
                    name="Filtered Incident / Box Laps",
                    marker=dict(size=11, color="rgba(255, 255, 0, 0.3)", symbol="triangle-up"),
                    text=["Yellow/Box" for _ in range(len(incident_laps))],
                    textposition="top center",
                    textfont=dict(size=10, color="rgba(255, 255, 100, 0.7)"),
                    hovertemplate="Filtered Incident Lap %{customdata}<extra>Yellow Flag</extra>",
                    customdata=incident_laps["lap_number"]
                ))

    # STAGE 4: FUEL MASS CORRECTION
    elif current_stage == 4:
        df_s4 = pd.DataFrame(stage_4) if stage_4 else pd.DataFrame()
        if not df_s4.empty:
            # Uncorrected raw times
            fig.add_trace(go.Scatter(
                x=df_s4["tyre_age_lap"],
                y=df_s4["lap_time_sec"],
                mode="markers",
                name="Heavy Fuel Lap Times",
                marker=dict(size=7, color="rgba(255, 255, 255, 0.35)", symbol="circle"),
                hovertemplate="Uncorrected: %{y:.3f}s (Fuel: %{text}kg)<extra></extra>",
                text=df_s4["fuel_kg"]
            ))
            # Fuel corrected times
            fig.add_trace(go.Scatter(
                x=df_s4["tyre_age_lap"],
                y=df_s4["fuel_corrected_time_sec"],
                mode="markers",
                name="Fuel-Corrected Pace (Dry Reference)",
                marker=dict(size=9, color="#00d2be", symbol="diamond"),
                hovertemplate="Fuel Corrected: %{y:.3f}s (-%{text:.2f}s fuel penalty)<extra></extra>",
                text=df_s4["fuel_penalty_sec"]
            ))
            # Adjustment connecting lines / arrows
            for _, r in df_s4.iterrows():
                fig.add_shape(
                    type="line",
                    x0=r["tyre_age_lap"],
                    y0=r["lap_time_sec"],
                    x1=r["tyre_age_lap"],
                    y1=r["fuel_corrected_time_sec"],
                    line=dict(color="rgba(0, 210, 190, 0.45)", width=1.5, dash="dot")
                )

    # STAGE 5: CLEAN DEGRADATION CURVE APPEARS
    elif current_stage == 5:
        four_tyres = pred_data.get("four_tyres", {})
        corner_curves = four_tyres.get("corner_gp_curves", noise_stages.get("corner_curves", {}))
        limiting = four_tyres.get("limiting_corner", "RL")

        corner_cfgs = {
            "FL": {"name": "FL Curve (Front-Left)", "color": "#00d2be"},
            "FR": {"name": "FR Curve (Front-Right)", "color": "#3498db"},
            "RL": {"name": "RL Curve (Rear-Left)", "color": "#ffd700"},
            "RR": {"name": "RR Curve (Rear-Right)", "color": "#ff793f"},
        }

        if corner_curves:
            if limiting in corner_curves and corner_curves[limiting]:
                df_lim = pd.DataFrame(corner_curves[limiting])
                fig.add_trace(go.Scatter(
                    x=list(df_lim["tyre_age"]) + list(df_lim["tyre_age"])[::-1],
                    y=list(df_lim["upper_bound_sec"]) + list(df_lim["lower_bound_sec"])[::-1],
                    fill="toself",
                    fillcolor="rgba(255, 24, 1, 0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name=f"{limiting} (Limiter) ±2σ Band"
                ))

            for corner, cfg in corner_cfgs.items():
                if corner in corner_curves and corner_curves[corner]:
                    df_c = pd.DataFrame(corner_curves[corner])
                    is_lim = (corner == limiting)
                    w = 3.8 if is_lim else 2.2
                    lbl = f"{cfg['name']} ★ LIMITER" if is_lim else cfg["name"]
                    fig.add_trace(go.Scatter(
                        x=df_c["tyre_age"],
                        y=df_c["predicted_lap_time_sec"],
                        mode="lines",
                        name=lbl,
                        line=dict(color=cfg["color"], width=w),
                        hovertemplate=f"<b>{corner}</b> (Lap %{{x}}): %{{y:.3f}}s<extra></extra>"
                    ))
        elif not df_curve.empty:
            fig.add_trace(go.Scatter(
                x=list(df_curve["tyre_age"]) + list(df_curve["tyre_age"])[::-1],
                y=list(df_curve["upper_bound_sec"]) + list(df_curve["lower_bound_sec"])[::-1],
                fill="toself",
                fillcolor="rgba(225, 6, 0, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="±2σ Gaussian Process Confidence Band"
            ))
            fig.add_trace(go.Scatter(
                x=df_curve["tyre_age"],
                y=df_curve["predicted_lap_time_sec"],
                mode="lines",
                name=f"Clean {compound} Degradation Curve",
                line=dict(color=comp_color, width=4),
                hovertemplate="Tyre Age %{x} Laps: %{y:.3f}s<extra>Predicted Pace</extra>"
            ))

        # 3. Clean Decoupled Stint Points
        df_s5 = pd.DataFrame(stage_5) if stage_5 else pd.DataFrame()
        if not df_s5.empty:
            fig.add_trace(go.Scatter(
                x=df_s5["tyre_age_lap"],
                y=df_s5["clean_lap_time_sec"],
                mode="markers",
                name="Decoupled Clean Telemetry",
                marker=dict(size=9, color="#00d2be", symbol="diamond", line=dict(width=1, color="#ffffff")),
                hovertemplate="Clean Pace: %{y:.3f}s (Age: %{x})<extra>Decoupled</extra>"
            ))

        # Annotate Pit & Cliff
        rec_pit = pred_data.get("recommended_pit_lap", 24)
        cliff = pred_data.get("cliff_lap", 26)
        fig.add_vline(x=rec_pit, line_width=1.8, line_dash="dash", line_color="#00d2be", annotation_text="Optimal Pit", annotation_position="top left")
        fig.add_vline(x=cliff, line_width=1.8, line_dash="dot", line_color="#ff1801", annotation_text="Tyre Cliff", annotation_position="top right")

    fig.update_layout(
        title=dict(
            text=f"{meta['title']}<br><sup style='color:#8c9ba5; font-size:11px;'>{meta['subtitle']}</sup>",
            font=dict(family="Orbitron", size=14, color="#ffffff")
        ),
        xaxis=dict(
            title="Tyre Age (Laps Completed on Compound)",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            dtick=2
        ),
        yaxis=dict(
            title="Lap Time (Seconds)",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False
        ),
        paper_bgcolor="rgba(11, 14, 20, 0.95)",
        plot_bgcolor="rgba(11, 14, 20, 0.95)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#8c9ba5")),
        hovermode="closest",
        height=480,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig


def create_side_by_side_comparison_chart(pred_data: Dict[str, Any]) -> go.Figure:
    """
    Creates a direct side-by-side comparison chart mirroring the reference image:
    Subplot 1: "With Outliers & Telemetry Noise" (distorted linear fit)
    Subplot 2: "Outliers Removed — A Much Better Fit!" (Gaussian Process clean fit)
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "With Outliers & Telemetry Noise",
            "Outliers Removed — A Much Better Fit!"
        ),
        horizontal_spacing=0.08
    )

    stages = pred_data.get("noise_removal_stages", {})
    compound = pred_data.get("compound", "MEDIUM").upper()
    comp_color = COMPOUND_COLORS.get(compound, "#ffd700")
    curve = pred_data.get("curve", [])
    df_curve = pd.DataFrame(curve) if curve else pd.DataFrame()
    stage_1 = stages.get("stage_1_raw", [])
    stage_5 = stages.get("stage_5_clean", [])
    naive_reg = stages.get("naive_regression", {"slope": 0.20, "intercept": 90.0})

    # ================= LEFT SUBPLOT: WITH OUTLIERS =================
    if stage_1:
        df_raw = pd.DataFrame(stage_1)
        # Red star / asterisk markers matching reference image
        fig.add_trace(
            go.Scatter(
                x=df_raw["tyre_age_lap"],
                y=df_raw["lap_time_sec"],
                mode="markers",
                name="Raw Observed Laps",
                marker=dict(size=9, color="#ff3333", symbol="star", line=dict(width=1, color="#ffffff")),
                hovertemplate="Lap %{text}: %{y:.3f}s (Age: %{x})<extra>Raw Lap</extra>",
                text=df_raw["lap_number"],
                showlegend=True
            ),
            row=1, col=1
        )

        # Distorted naive trendline (blue dashed line matching reference image)
        min_x, max_x = df_raw["tyre_age_lap"].min(), df_raw["tyre_age_lap"].max()
        x_line = np.linspace(min_x, max_x, 25)
        y_line = naive_reg["intercept"] + naive_reg["slope"] * x_line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Naive Skewed Fit",
                line=dict(color="#3b82f6", width=2.5, dash="dash"),
                hovertemplate="Naive Fit: %{y:.3f}s<extra></extra>",
                showlegend=True
            ),
            row=1, col=1
        )

    # ================= RIGHT SUBPLOT: OUTLIERS REMOVED =================
    four_tyres = pred_data.get("four_tyres", {})
    corner_curves = four_tyres.get("corner_gp_curves", {})
    limiting = four_tyres.get("limiting_corner", "RL")

    corner_cfgs = {
        "FL": {"name": "FL (Front-Left)", "color": "#00d2be"},
        "FR": {"name": "FR (Front-Right)", "color": "#3498db"},
        "RL": {"name": "RL (Rear-Left)", "color": "#ffd700"},
        "RR": {"name": "RR (Rear-Right)", "color": "#ff793f"},
    }

    if corner_curves:
        if limiting in corner_curves and corner_curves[limiting]:
            df_lim = pd.DataFrame(corner_curves[limiting])
            fig.add_trace(
                go.Scatter(
                    x=list(df_lim["tyre_age"]) + list(df_lim["tyre_age"])[::-1],
                    y=list(df_lim["upper_bound_sec"]) + list(df_lim["lower_bound_sec"])[::-1],
                    fill="toself",
                    fillcolor="rgba(255, 24, 1, 0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name=f"{limiting} ±2σ Band",
                    showlegend=True
                ),
                row=1, col=2
            )

        for corner, cfg in corner_cfgs.items():
            if corner in corner_curves and corner_curves[corner]:
                df_c = pd.DataFrame(corner_curves[corner])
                is_lim = (corner == limiting)
                w = 3.6 if is_lim else 2.2
                name_l = f"{cfg['name']} ★ LIMITER" if is_lim else cfg["name"]
                fig.add_trace(
                    go.Scatter(
                        x=df_c["tyre_age"],
                        y=df_c["predicted_lap_time_sec"],
                        mode="lines",
                        name=name_l,
                        line=dict(color=cfg["color"], width=w),
                        hovertemplate=f"<b>{corner}</b> (Lap %{{x}}): %{{y:.3f}}s<extra></extra>",
                        showlegend=True
                    ),
                    row=1, col=2
                )
    elif not df_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=list(df_curve["tyre_age"]) + list(df_curve["tyre_age"])[::-1],
                y=list(df_curve["upper_bound_sec"]) + list(df_curve["lower_bound_sec"])[::-1],
                fill="toself",
                fillcolor="rgba(225, 6, 0, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="±2σ Confidence Band",
                showlegend=True
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=df_curve["tyre_age"],
                y=df_curve["predicted_lap_time_sec"],
                mode="lines",
                name=f"True {compound} Curve",
                line=dict(color=comp_color, width=3.5),
                hovertemplate="Tyre Age %{x}: %{y:.3f}s<extra>Clean Pace</extra>",
                showlegend=True
            ),
            row=1, col=2
        )

    # 3. Clean points
    if stage_5:
        df_clean = pd.DataFrame(stage_5)
        fig.add_trace(
            go.Scatter(
                x=df_clean["tyre_age_lap"],
                y=df_clean["clean_lap_time_sec"],
                mode="markers",
                name="Decoupled Clean Laps",
                marker=dict(size=8, color="#00d2be", symbol="diamond", line=dict(width=1, color="#ffffff")),
                hovertemplate="Clean Lap: %{y:.3f}s (Age: %{x})<extra>Clean</extra>",
                showlegend=True
            ),
            row=1, col=2
        )

    # Layout styling
    fig.update_layout(
        paper_bgcolor="rgba(11, 14, 20, 0.95)",
        plot_bgcolor="rgba(11, 14, 20, 0.95)",
        font=dict(color="#8c9ba5"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        height=460,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.update_xaxes(title_text="Tyre Age (Laps Completed)", gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_xaxes(title_text="Tyre Age (Laps Completed)", gridcolor="rgba(255,255,255,0.06)", row=1, col=2)
    fig.update_yaxes(title_text="Lap Time (Seconds)", gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(title_text="Lap Time (Seconds)", gridcolor="rgba(255,255,255,0.06)", row=1, col=2)

    return fig
