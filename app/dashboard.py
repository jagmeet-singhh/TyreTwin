import streamlit as st
from app.utils.styles import apply_f1_theme
from app.components.selectors import render_f1_sidebar
from app.components.kpi_cards import render_metric_card
from app.components.confidence_gauge import render_confidence_gauge
from app.components.strategy_card import render_strategy_card
from app.charts.degradation_plots import create_degradation_chart
from app.charts.four_wheel_plots import create_four_wheel_degradation_chart
from app.components.car_schematic import render_four_wheel_car_schematic
from app.utils.api_client import TyreTwinClient
from app.charts.noise_removal_plots import create_noise_removal_stage_chart, create_side_by_side_comparison_chart
import time

st.set_page_config(
    page_title="TyreTwin // F1 Pit Wall Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_f1_theme()

# Sidebar Selectors
season, grand_prix, session_name, driver_code, compound = render_f1_sidebar()
session_id = f"{season}_{grand_prix}_{session_name}"

# Header Banner
st.markdown(
    f"""
    <div class="f1-header-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 1.8rem; color: #ffffff;">TYRETWIN // PIT WALL COMMAND CENTER</h1>
                <div style="color: #8c9ba5; font-size: 0.95rem; margin-top: 4px;">
                    FORMULA 1 AI TELEMETRY DECOUPLING & FOUR-CORNER TYRE DEGRADATION INTELLIGENCE
                </div>
            </div>
            <div style="text-align: right;">
                <span class="compound-badge compound-{compound.lower()}">{compound} TYRE</span>
                <div style="color: #00d2be; font-family: 'Orbitron', sans-serif; font-size: 0.85rem; margin-top: 4px;">
                    LIVE TELEMETRY SYNCED
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Fetch Data from API Client
with st.spinner("Analyzing high-frequency telemetry and decoupling noise..."):
    sess_info = TyreTwinClient.load_session(season, grand_prix, session_name)
    pred_data = TyreTwinClient.predict_degradation(session_id, driver_code, compound)
    strat_data = TyreTwinClient.get_strategy(session_id, driver_code, current_lap=10, current_compound=compound, tyre_age=10)

four_tyres = pred_data.get("four_tyres", {})
limiting_corner = four_tyres.get("limiting_corner", "RL") if four_tyres else "RL"
limiting_name = four_tyres.get("limiting_corner_name", "Rear-Left") if four_tyres else "Rear-Left"

# Executive KPI Metric Cards
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    render_metric_card(
        label="REMAINING TYRE LIFE",
        value=f"{pred_data.get('remaining_life_laps', 14)} LAPS",
        delta=f"Cliff: Lap {pred_data.get('cliff_lap', 26)}",
        delta_color="#ff1801" if pred_data.get('remaining_life_laps', 14) < 6 else "#00d2be",
        subtext=f"Limiter: {limiting_corner} ({limiting_name})"
    )

with c2:
    render_metric_card(
        label="OPTIMAL PIT WINDOW",
        value=f"LAP {pred_data.get('recommended_pit_lap', 24)}",
        delta=f"Window: L{pred_data.get('pit_window_start', 22)}-L{pred_data.get('pit_window_end', 26)}",
        delta_color="#00d2be",
        subtext="Target Compound: HARD"
    )

with c3:
    render_metric_card(
        label="AVG WEAR GRADIENT",
        value=f"{pred_data.get('degradation_slope_sec_per_lap', 0.054):.3f}s",
        delta=f"{limiting_corner} Pace Falloff",
        delta_color="#ffd700",
        subtext="Pace drop per lap"
    )

with c4:
    render_metric_card(
        label="BASE CLEAN PACE",
        value=f"{pred_data.get('base_clean_lap_sec', 91.5):.3f}s",
        delta="-0.450s Noise Decoupled",
        delta_color="#00d2be",
        subtext=f"Circuit: {sess_info.get('track_name', 'Bahrain')}"
    )

with c5:
    render_confidence_gauge(pred_data.get("confidence_pct", 94.5))

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Main Grid: Left Chart + Right Strategy Panel
col_chart, col_side = st.columns([2.2, 1.0])

with col_chart:
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.05rem; font-weight: 700; color: #ffffff;">
                LIVE TELEMETRY NOISE DECOUPLING & TYRE INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    view_mode = st.radio(
        "Visualization Mode",
        [
            "🏎️ 4-Corner Car Schematic (FL, FR, RL, RR)",
            "5-Stage Noise Removal Pipeline",
            "Side-by-Side Comparison (With vs Without Outliers)",
            "Standard Pit Wall Pace Curve"
        ],
        horizontal=True,
        index=0,
        label_visibility="collapsed"
    )

    if view_mode == "🏎️ 4-Corner Car Schematic (FL, FR, RL, RR)":
        render_four_wheel_car_schematic(four_tyres)
        st.markdown("<div style='margin-top: 28px; margin-bottom: 14px; border-top: 1px solid rgba(255, 255, 255, 0.1);'></div>", unsafe_allow_html=True)
        fig_4deg = create_four_wheel_degradation_chart(four_tyres)
        st.plotly_chart(fig_4deg, use_container_width=True)

    elif view_mode == "5-Stage Noise Removal Pipeline":
        # Interactive 5-stage stepper controls
        stage_col1, stage_col2 = st.columns([3, 1])
        
        with stage_col1:
            stage_labels = [
                "1. Raw Practice Laps",
                "2. Traffic Disappears",
                "3. Yellow Flags Disappear",
                "4. Fuel Correction",
                "5. Clean Curve Appears"
            ]
            selected_stage_label = st.select_slider(
                "Telemetry Decoupling Stage",
                options=stage_labels,
                value=stage_labels[0],
                label_visibility="collapsed"
            )
            stage_idx = stage_labels.index(selected_stage_label) + 1

        with stage_col2:
            auto_play = st.button("▶ Auto-Play Pipeline", use_container_width=True)

        chart_placeholder = st.empty()

        if auto_play:
            for s in range(1, 6):
                with chart_placeholder.container():
                    fig_s = create_noise_removal_stage_chart(pred_data, current_stage=s)
                    st.plotly_chart(fig_s, use_container_width=True, key=f"auto_stage_{s}")
                time.sleep(1.2)
        else:
            fig_stage = create_noise_removal_stage_chart(pred_data, current_stage=stage_idx)
            chart_placeholder.plotly_chart(fig_stage, use_container_width=True)

        # Stage Telemetry Diagnostic Metrics
        stage_counts = pred_data.get("noise_removal_stages", {}).get("counts", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("1. Raw Laps", f"{stage_counts.get('raw', 15)} Laps", "Unfiltered")
        m2.metric("2. Traffic Removed", f"-{stage_counts.get('traffic_removed', 2)} Laps", "+0.85s noise dropped", delta_color="inverse")
        m3.metric("3. Incidents Removed", f"-{stage_counts.get('incidents_removed', 2)} Laps", "Yellow/Box dropped", delta_color="inverse")
        m4.metric("4. Fuel Burn Offset", "-0.30s / 10kg", "Mass compensation", delta_color="normal")
        m5.metric("5. Clean Fitted Laps", f"{stage_counts.get('after_fuel', 11)} Laps", "GP ±2σ Modelled", delta_color="normal")

    elif view_mode == "Side-by-Side Comparison (With vs Without Outliers)":
        st.caption("Direct telemetry comparison matching outlier removal and Gaussian Process degradation modeling:")
        fig_sbs = create_side_by_side_comparison_chart(pred_data)
        st.plotly_chart(fig_sbs, use_container_width=True)

    else:
        fig_deg = create_degradation_chart(pred_data)
        st.plotly_chart(fig_deg, use_container_width=True)

with col_side:
    st.subheader("Live Strategy Directive")
    render_strategy_card(strat_data)
    
    alert_html = (
        f'<div style="background: rgba(20, 24, 33, 0.85); border: 1px solid rgba(225, 6, 0, 0.25); border-radius: 8px; padding: 14px; margin-top: 12px;">'
        f'<div style="font-family: \'Orbitron\', sans-serif; font-size: 0.9rem; font-weight: 700; color: #ffffff;">4-Wheel Corner Alert</div>'
        f'<div style="font-size: 0.85rem; color: #8c9ba5; margin-top: 6px;">'
        f'• Limiting Corner: <b style="color: #ff1801;">{limiting_corner} ({limiting_name})</b><br>'
        f'• Wear Limiter Cliff: <b style="color: #ffd700;">Lap {pred_data.get("cliff_lap", 24)}</b><br>'
        f'• Asymmetry: <b style="color: #00d2be;">{four_tyres.get("circuit_limitation", "Balanced")}</b>'
        f'</div></div>'
    )
    st.markdown(alert_html, unsafe_allow_html=True)
