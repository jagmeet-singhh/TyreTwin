import streamlit as st
import plotly.graph_objects as go
from app.utils.styles import apply_f1_theme
from app.components.selectors import render_f1_sidebar
from app.components.strategy_card import render_strategy_card
from app.utils.api_client import TyreTwinClient

st.set_page_config(page_title="Strategy Assistant // TyreTwin", layout="wide")
apply_f1_theme()

season, grand_prix, session_name, driver_code, compound = render_f1_sidebar()
session_id = f"{season}_{grand_prix}_{session_name}"

st.markdown(
    """
    <div class="f1-header-container">
        <h1 style="margin:0; color:#ffffff; font-size:1.6rem;">PIT WALL STRATEGY ASSISTANT</h1>
        <div style="color:#8c9ba5; font-size:0.9rem; margin-top:4px;">
            DYNAMIC RACE SIMULATION, PIT WINDOW SOLVER & UNDERCUT DELTAS
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)
with c1:
    curr_lap = st.slider("Current Race Lap", 1, 57, 12)
with c2:
    tyre_age = st.slider("Current Tyre Age (Laps)", 1, 35, 12)

strat_data = TyreTwinClient.get_strategy(session_id, driver_code, curr_lap, compound, tyre_age)

render_strategy_card(strat_data)

# Strategy Comparison Simulation Plot
st.subheader("1-Stop vs 2-Stop Race Pace Projection")
curve = strat_data.get("pace_comparison_curve", [])
if curve:
    import pandas as pd
    df = pd.DataFrame(curve)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["lap"], y=df["strategy_1stop_pace_sec"], name="1-Stop (Plan A)", line=dict(color="#00d2be", width=2.5)))
    fig.add_trace(go.Scatter(x=df["lap"], y=df["strategy_2stop_pace_sec"], name="2-Stop (Plan B)", line=dict(color="#e10600", width=2, dash="dash")))
    fig.update_layout(
        xaxis=dict(title="Race Lap", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Projected Lap Time (s)", gridcolor="rgba(255,255,255,0.06)"),
        paper_bgcolor="rgba(11, 14, 20, 0.95)",
        plot_bgcolor="rgba(11, 14, 20, 0.95)",
        font=dict(color="#8c9ba5"),
        height=360,
        margin=dict(l=40, r=40, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
