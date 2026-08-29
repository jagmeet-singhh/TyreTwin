import streamlit as st
from app.utils.styles import apply_f1_theme
from app.components.selectors import render_f1_sidebar
from app.charts.degradation_plots import create_degradation_chart
from app.charts.shap_plots import create_shap_waterfall_chart
from app.utils.api_client import TyreIQClient

st.set_page_config(page_title="Degradation Intelligence // TyreIQ", page_icon="📈", layout="wide")
apply_f1_theme()

season, grand_prix, session_name, driver_code, compound = render_f1_sidebar()
session_id = f"{season}_{grand_prix}_{session_name}"

st.markdown(
    """
    <div class="f1-header-container">
        <h1 style="margin:0; color:#ffffff; font-size:1.6rem;">📈 TYRE DEGRADATION INTELLIGENCE</h1>
        <div style="color:#8c9ba5; font-size:0.9rem; margin-top:4px;">
            GAUSSIAN PROCESS PROBABILISTIC MODELLING & TELEMETRY NOISE ISOLATION
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

stint_len = st.slider("Forecast Stint Length (Laps)", min_value=10, max_value=45, value=28, step=1)

with st.spinner("Generating Gaussian Process Posterior with ±2σ Confidence Intervals..."):
    pred_data = TyreIQClient.predict_degradation(session_id, driver_code, compound, stint_length=stint_len)

col_main, col_shap = st.columns([1.8, 1.2])

with col_main:
    fig_deg = create_degradation_chart(pred_data)
    st.plotly_chart(fig_deg, use_container_width=True)

with col_shap:
    st.subheader("🔍 Explainable AI (SHAP)")
    st.caption("External noise decomposition for selected stint telemetry:")
    attrs = pred_data.get("feature_attributions", [])
    fig_shap = create_shap_waterfall_chart(attrs)
    st.plotly_chart(fig_shap, use_container_width=True)

# Data Table Expander
with st.expander("📊 View Model Prediction Stint Data Table"):
    st.dataframe(pred_data.get("curve", []), use_container_width=True)
