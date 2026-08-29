import streamlit as st
from app.utils.styles import apply_f1_theme
from app.components.selectors import render_f1_sidebar
from app.utils.pdf_generator import PDFReportGenerator
from app.utils.api_client import TyreIQClient

st.set_page_config(page_title="Engineering Report // TyreIQ", page_icon="📑", layout="wide")
apply_f1_theme()

season, grand_prix, session_name, driver_code, compound = render_f1_sidebar()
session_id = f"{season}_{grand_prix}_{session_name}"

st.markdown(
    """
    <div class="f1-header-container">
        <h1 style="margin:0; color:#ffffff; font-size:1.6rem;">📑 PIT WALL ENGINEERING DEBRIEF REPORT</h1>
        <div style="color:#8c9ba5; font-size:0.9rem; margin-top:4px;">
            ONE-CLICK EXECUTIVE PDF GENERATION FOR RACE ENGINEERS & CHIEF STRATEGISTS
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

pred_data = TyreIQClient.predict_degradation(session_id, driver_code, compound)

st.write("Ready to compile official session debrief sheet containing degradation coefficients, telemetry metrics, and strategy directives.")

pdf_bytes = PDFReportGenerator.create_pitwall_report(pred_data)

st.download_button(
    label="📥 Download Official Pit Wall Report (PDF)",
    data=pdf_bytes,
    file_name=f"TyreIQ_Debrief_{driver_code}_{compound}_{session_id}.pdf",
    mime="application/pdf",
    use_container_width=True
)

st.info("💡 Report contains Executive Summary, Gaussian Process Parameters, ±2σ Uncertainty Thresholds, and Decoupled Noise Attributions.")
