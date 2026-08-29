import streamlit as st
from app.utils.styles import apply_f1_theme
from app.components.selectors import render_f1_sidebar
from app.charts.telemetry_plots import create_telemetry_multitrace_chart
from app.utils.api_client import TyreIQClient

st.set_page_config(page_title="Telemetry Explorer // TyreIQ", page_icon="🏎️", layout="wide")
apply_f1_theme()

season, grand_prix, session_name, driver_code, compound = render_f1_sidebar()
session_id = f"{season}_{grand_prix}_{session_name}"

st.markdown(
    """
    <div class="f1-header-container">
        <h1 style="margin:0; color:#ffffff; font-size:1.6rem;">🏎️ HIGH-FREQUENCY TELEMETRY EXPLORER</h1>
        <div style="color:#8c9ba5; font-size:0.9rem; margin-top:4px;">
            SYNCHRONIZED MULTI-SENSOR TRACE ANALYSIS ACROSS STINT LAPS
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2 = st.columns([1, 3])
with c1:
    lap_num = st.selectbox("Select Lap to Inspect", [3, 5, 8, 12, 16, 20], index=1)

with st.spinner("Fetching synchronized 100Hz telemetry channels..."):
    tel_data = TyreIQClient.get_telemetry(session_id, driver_code, lap_num)

fig_tel = create_telemetry_multitrace_chart(tel_data)
st.plotly_chart(fig_tel, use_container_width=True)
