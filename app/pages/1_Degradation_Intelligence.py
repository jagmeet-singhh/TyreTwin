import streamlit as st
from app.utils.styles import apply_f1_theme
from app.components.selectors import render_f1_sidebar
from app.charts.degradation_plots import create_degradation_chart
from app.charts.shap_plots import create_shap_waterfall_chart
from app.charts.four_wheel_plots import create_four_wheel_degradation_chart, create_four_wheel_thermal_chart
from app.components.car_schematic import render_four_wheel_car_schematic
from app.utils.api_client import TyreTwinClient
import time
import pandas as pd
from app.charts.noise_removal_plots import create_noise_removal_stage_chart, create_side_by_side_comparison_chart

st.set_page_config(page_title="Degradation Intelligence // TyreTwin", layout="wide")
apply_f1_theme()

season, grand_prix, session_name, driver_code, compound = render_f1_sidebar()
session_id = f"{season}_{grand_prix}_{session_name}"

st.markdown(
    """
    <div class="f1-header-container">
        <h1 style="margin:0; color:#ffffff; font-size:1.6rem;">TYRE DEGRADATION INTELLIGENCE</h1>
        <div style="color:#8c9ba5; font-size:0.9rem; margin-top:4px;">
            FOUR-CORNER (FL, FR, RL, RR) DEGRADATION & GAUSSIAN PROCESS TELEMETRY DECOUPLING
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

stint_len = st.slider("Forecast Stint Length (Laps)", min_value=10, max_value=45, value=28, step=1)

with st.spinner("Analyzing 4-wheel tyre telemetry and generating Gaussian Process posterior..."):
    pred_data = TyreTwinClient.predict_degradation(session_id, driver_code, compound, stint_length=stint_len)

four_tyres = pred_data.get("four_tyres", {})

# Mode Selector Tabs
tab_4w, tab_pipeline, tab_sbs, tab_shap = st.tabs([
    "🏎️ 4-Corner Tyre Dynamics (FL, FR, RL, RR)",
    "Interactive 5-Stage Decoupling Pipeline",
    "Side-by-Side Comparison (With vs Outliers Removed)",
    "SHAP Telemetry Attribution"
])

with tab_4w:
    st.caption("Individual tyre wear, load distribution, and thermal degradation across all 4 wheels:")
    render_four_wheel_car_schematic(four_tyres)
    st.markdown("<div style='margin-top: 28px; margin-bottom: 14px; border-top: 1px solid rgba(255, 255, 255, 0.1);'></div>", unsafe_allow_html=True)
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        fig_4deg = create_four_wheel_degradation_chart(four_tyres)
        st.plotly_chart(fig_4deg, use_container_width=True)
    with c_chart2:
        fig_4th = create_four_wheel_thermal_chart(four_tyres)
        st.plotly_chart(fig_4th, use_container_width=True)

    # 4-Corner Stint Data Matrix
    with st.expander("📊 View 4-Corner Lap-by-Lap Degradation Matrix"):
        hist = four_tyres.get("history", [])
        if hist:
            df_hist = pd.DataFrame(hist)
            st.dataframe(df_hist, use_container_width=True)

with tab_pipeline:
    st.caption("Step through the 5 motorsport physics filters decoupling external telemetry noise:")
    
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        stage_names = [
            "1. Raw Practice Laps",
            "2. Traffic Laps Disappear",
            "3. Yellow Flag Laps Disappear",
            "4. Fuel Mass Correction",
            "5. Clean Degradation Curve"
        ]
        chosen_stage = st.select_slider("Select Pipeline Stage", options=stage_names, value=stage_names[0])
        stage_num = stage_names.index(chosen_stage) + 1
    
    with col_ctrl2:
        st.write("")
        play_btn = st.button("▶ Run Live Animation", use_container_width=True, key="deg_page_play")

    stage_container = st.empty()
    if play_btn:
        for stg in range(1, 6):
            with stage_container.container():
                f_s = create_noise_removal_stage_chart(pred_data, current_stage=stg)
                st.plotly_chart(f_s, use_container_width=True, key=f"deg_auto_{stg}")
            time.sleep(1.2)
    else:
        f_stg = create_noise_removal_stage_chart(pred_data, current_stage=stage_num)
        stage_container.plotly_chart(f_stg, use_container_width=True)

    # Diagnostic KPI summary
    stage_counts = pred_data.get("noise_removal_stages", {}).get("counts", {})
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Stage 1: Raw Telemetry", f"{stage_counts.get('raw', 15)} Laps", "Full session scatter")
    k2.metric("Stage 2: Traffic Filtered", f"-{stage_counts.get('traffic_removed', 2)} Laps", "Dirty air excluded", delta_color="inverse")
    k3.metric("Stage 3: Incidents Filtered", f"-{stage_counts.get('incidents_removed', 2)} Laps", "Yellow/Box excluded", delta_color="inverse")
    k4.metric("Stage 4: Fuel Corrected", "-0.30s / 10kg", "Standard dry baseline", delta_color="normal")
    k5.metric("Stage 5: Final GP Model", f"{stage_counts.get('after_fuel', 11)} Laps", "±2σ Bound Fitted", delta_color="normal")

with tab_sbs:
    st.caption("Direct telemetry comparison matching the reference illustration (left: uncorrected outliers; right: decoupled clean curve):")
    fig_comp = create_side_by_side_comparison_chart(pred_data)
    st.plotly_chart(fig_comp, use_container_width=True)

with tab_shap:
    col_main, col_shap = st.columns([1.6, 1.4])
    with col_main:
        fig_deg = create_degradation_chart(pred_data)
        st.plotly_chart(fig_deg, use_container_width=True)
    with col_shap:
        st.subheader("Explainable AI (SHAP)")
        st.caption("External noise decomposition for selected stint telemetry:")
        attrs = pred_data.get("feature_attributions", [])
        fig_shap = create_shap_waterfall_chart(attrs)
        st.plotly_chart(fig_shap, use_container_width=True)

# Data Table Expander
with st.expander("View Gaussian Process Fitted Stint Data Table"):
    st.dataframe(pred_data.get("curve", []), use_container_width=True)
