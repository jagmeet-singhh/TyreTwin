import streamlit as st
from typing import Dict, Any


def render_strategy_card(strategy_data: Dict[str, Any]):
    opt_strat = strategy_data.get("recommended_strategy", {})
    name = opt_strat.get("strategy_name", "1-Stop Plan A")
    pit_window = opt_strat.get("pit_window", "Laps 22-26")
    cliff = opt_strat.get("tyre_cliff_warning_lap", 26)
    summary = opt_strat.get("recommendation_summary", "")

    html = f"""
    <div class="strategy-box">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-family:'Orbitron', sans-serif; font-size:1.1rem; font-weight:800; color:#ffffff;">
                🎯 {name}
            </div>
            <span class="compound-badge compound-medium">OPTIMAL</span>
        </div>
        <div style="margin-top:8px; font-size:0.92rem; color:#cbd5e0; line-height:1.4;">
            {summary}
        </div>
        <div style="display:flex; gap:24px; margin-top:12px; font-family:'Orbitron', sans-serif; font-size:0.85rem;">
            <div><span style="color:#8c9ba5;">PIT WINDOW:</span> <span style="color:#00d2be; font-weight:700;">{pit_window}</span></div>
            <div><span style="color:#8c9ba5;">TYRE CLIFF:</span> <span style="color:#ff1801; font-weight:700;">Lap {cliff}</span></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
