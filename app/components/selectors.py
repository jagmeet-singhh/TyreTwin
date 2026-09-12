import streamlit as st
from typing import Tuple, Dict, Any
from ml.fastf1_loader import BENCHMARK_CONFIGS, DRIVERS_METADATA


def render_f1_sidebar() -> Tuple[int, str, str, str, str]:
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0 18px 0;">
            <span style="font-family: 'Orbitron', sans-serif; font-size: 1.6rem; font-weight: 900; color: #e10600; letter-spacing: 2px;">TYRE</span>
            <span style="font-family: 'Orbitron', sans-serif; font-size: 1.6rem; font-weight: 900; color: #ffffff; letter-spacing: 2px;">TWIN</span>
            <div style="font-size: 0.72rem; color: #8c9ba5; letter-spacing: 1px; text-transform: uppercase; margin-top: 2px;">Pit Wall Telemetry AI</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Control")
    
    season = st.sidebar.selectbox("Season", [2024, 2023], index=0)
    grand_prix = st.sidebar.selectbox(
        "Grand Prix", ["Bahrain", "Silverstone", "Monza", "Suzuka"], index=0
    )
    session_name = st.sidebar.selectbox(
        "Session", ["Qualifying", "FP2", "FP1", "FP3", "Race"], index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Driver & Tyre")
    
    driver_options = {d["code"]: f"{d['number']} - {d['full_name']} ({d['team_name']})" for d in DRIVERS_METADATA}
    driver_code = st.sidebar.selectbox(
        "Driver",
        options=list(driver_options.keys()),
        format_func=lambda x: driver_options[x],
        index=0
    )

    compound = st.sidebar.selectbox(
        "Tyre Compound",
        ["MEDIUM", "SOFT", "HARD", "INTERMEDIATE", "WET"],
        index=0
    )

    # Color badge preview
    badge_class = f"compound-{compound.lower()}"
    st.sidebar.markdown(
        f'<div style="text-align:center; margin-top:10px;"><span class="compound-badge {badge_class}">{compound} TYRE</span></div>',
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("TyreTwin v1.0 • Connected to FastF1")

    return season, grand_prix, session_name, driver_code, compound
