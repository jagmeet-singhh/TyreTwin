import streamlit as st
import plotly.graph_objects as go


def render_confidence_gauge(confidence_pct: float = 94.5):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "AI Confidence", "font": {"size": 14, "color": "#8c9ba5", "family": "Titillium Web"}},
        number={"suffix": "%", "font": {"size": 24, "color": "#ffffff", "family": "Orbitron"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a5568"},
            "bar": {"color": "#e10600"},
            "bgcolor": "rgba(20, 24, 33, 0.9)",
            "borderwidth": 1,
            "bordercolor": "rgba(225, 6, 0, 0.3)",
            "steps": [
                {"range": [0, 60], "color": "rgba(255, 24, 1, 0.2)"},
                {"range": [60, 85], "color": "rgba(255, 215, 0, 0.2)"},
                {"range": [85, 100], "color": "rgba(0, 210, 190, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "#00d2be", "width": 3},
                "thickness": 0.75,
                "value": confidence_pct
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=25, b=15),
        height=140,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
