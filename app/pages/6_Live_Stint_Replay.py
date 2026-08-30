import streamlit as st
import time
import plotly.graph_objects as go
from app.utils.styles import apply_f1_theme
from app.components.selectors import render_f1_sidebar
from app.components.kpi_cards import render_metric_card
from app.utils.api_client import TyreTwinClient

st.set_page_config(page_title="Live Stint Replay // TyreTwin", layout="wide")
apply_f1_theme()

season, grand_prix, session_name, driver_code, compound = render_f1_sidebar()
session_id = f"{season}_{grand_prix}_{session_name}"

st.markdown(
    """
    <div class="f1-header-container">
        <h1 style="margin:0; color:#ffffff; font-size:1.6rem;">LIVE STINT REPLAY & TELEMETRY STREAM</h1>
        <div style="color:#8c9ba5; font-size:0.9rem; margin-top:4px;">
            REAL-TIME TURN-BY-TURN TELEMETRY PLAYBACK WITH NOISE DECOUPLING
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

replay_lap = st.slider("Scrub Stint Lap", min_value=1, max_value=24, value=8)

tel = TyreTwinClient.get_telemetry(session_id, driver_code, replay_lap)

c1, c2, c3 = st.columns(3)
with c1:
    render_metric_card("CURRENT LAP TIME", f"{tel.get('lap_time_sec', 91.5):.3f}s", "Raw Telemetry", "#ffffff")
with c2:
    render_metric_card("CLEAN LAP TIME", f"{tel.get('clean_lap_time_sec', 91.1):.3f}s", "-0.40s Noise Decoupled", "#00d2be")
with c3:
    render_metric_card("TELEMETRY NOISE", f"+{tel.get('noise_sec', 0.40):.3f}s", "Fuel + Traffic + Grip", "#ffd700")

st.caption("Live high-frequency throttle and speed traces:")
from app.charts.telemetry_plots import create_telemetry_multitrace_chart
st.plotly_chart(create_telemetry_multitrace_chart(tel), use_container_width=True)
