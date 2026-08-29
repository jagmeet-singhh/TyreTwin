import streamlit as st
from app.utils.styles import apply_f1_theme
from app.components.selectors import render_f1_sidebar
from app.components.kpi_cards import render_metric_card
from app.components.confidence_gauge import render_confidence_gauge
from app.components.strategy_card import render_strategy_card
from app.charts.degradation_plots import create_degradation_chart
from app.utils.api_client import TyreIQClient

st.set_page_config(
    page_title="TyreIQ // F1 Pit Wall Intelligence",
    page_icon="🏎️",
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
                <h1 style="margin: 0; font-size: 1.8rem; color: #ffffff;">🏎️ TYREIQ // PIT WALL COMMAND CENTER</h1>
                <div style="color: #8c9ba5; font-size: 0.95rem; margin-top: 4px;">
                    FORMULA 1 AI TELEMETRY DECOUPLING & TYRE DEGRADATION INTELLIGENCE
                </div>
            </div>
            <div style="text-align: right;">
                <span class="compound-badge compound-{compound.lower()}">{compound} TYRE</span>
                <div style="color: #00d2be; font-family: 'Orbitron', sans-serif; font-size: 0.85rem; margin-top: 4px;">
                    ● LIVE TELEMETRY SYNCED
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Fetch Data from API Client
with st.spinner("Analyzing high-frequency telemetry and decoupling noise..."):
    sess_info = TyreIQClient.load_session(season, grand_prix, session_name)
    pred_data = TyreIQClient.predict_degradation(session_id, driver_code, compound)
    strat_data = TyreIQClient.get_strategy(session_id, driver_code, current_lap=10, current_compound=compound, tyre_age=10)

# Executive KPI Metric Cards
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    render_metric_card(
        label="REMAINING TYRE LIFE",
        value=f"{pred_data.get('remaining_life_laps', 14)} LAPS",
        delta=f"Cliff: Lap {pred_data.get('cliff_lap', 26)}",
        delta_color="#ff1801" if pred_data.get('remaining_life_laps', 14) < 6 else "#00d2be",
        subtext=f"Current Compound: {compound}"
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
        delta="+0.012s vs FP1",
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
    fig_deg = create_degradation_chart(pred_data)
    st.plotly_chart(fig_deg, use_container_width=True)

with col_side:
    st.subheader("🎯 Live Strategy Directive")
    render_strategy_card(strat_data)
    
    st.markdown(
        """
        <div style="background: rgba(20, 24, 33, 0.85); border: 1px solid rgba(225, 6, 0, 0.25); border-radius: 8px; padding: 14px; margin-top: 12px;">
            <div style="font-family: 'Orbitron', sans-serif; font-size: 0.9rem; font-weight: 700; color: #ffffff;">
                🌡️ Track & Thermal Status
            </div>
            <div style="font-size: 0.85rem; color: #8c9ba5; margin-top: 6px;">
                • Track Temp: <b style="color: #ffffff;">33.8°C</b> (Optimal)<br>
                • Air Temp: <b style="color: #ffffff;">24.5°C</b> | Humidity: <b style="color: #ffffff;">42%</b><br>
                • Track Evolution: <b style="color: #00d2be;">+0.35s grip gain</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
