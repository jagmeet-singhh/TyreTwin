import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    delta: str = "",
    delta_color: str = "#00d2be",
    subtext: str = "",
):
    delta_html = f'<div class="f1-metric-delta" style="color: {delta_color};">{delta}</div>' if delta else ""
    sub_html = f'<div style="color: #8c9ba5; font-size: 0.78rem; margin-top: 2px;">{subtext}</div>' if subtext else ""
    
    html = f"""
    <div class="f1-metric-card">
        <div class="f1-metric-label">{label}</div>
        <div class="f1-metric-value">{value}</div>
        {delta_html}
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
