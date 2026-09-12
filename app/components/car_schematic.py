import streamlit as st
from typing import Dict, Any, Optional


def render_four_wheel_car_schematic(four_tyres_data: Optional[Dict[str, Any]]):
    """
    Renders an interactive high-tech Formula 1 four-wheel telemetry matrix using native Streamlit columns:
    - Row 1: Front-Left (FL) and Front-Right (FR)
    - Row 2: Rear-Left (RL) and Rear-Right (RR)
    Uses native document flow so the graph below is pushed down and never covers the rear wheels.
    """
    if not four_tyres_data or "tyres" not in four_tyres_data:
        st.info("4-Wheel Telemetry currently synchronizing...")
        return

    tyres = four_tyres_data.get("tyres", {})
    limiting = four_tyres_data.get("limiting_corner", "RL")
    circuit_name = four_tyres_data.get("circuit_name", "Grand Prix Circuit")
    limitation_text = four_tyres_data.get("circuit_limitation", "Asymmetric Wear")
    desc = four_tyres_data.get("circuit_description", "")
    pit_lap = four_tyres_data.get("recommended_pit_lap", 24)

    fl = tyres.get("FL", {})
    fr = tyres.get("FR", {})
    rl = tyres.get("RL", {})
    rr = tyres.get("RR", {})

    def get_bar_color(pct: float) -> str:
        if pct >= 70:
            return "#00d2be"
        elif pct >= 45:
            return "#ffd700"
        elif pct >= 25:
            return "#ff793f"
        else:
            return "#ff1801"

    def corner_card_html(c_data: Dict[str, Any], corner_id: str) -> str:
        name = c_data.get("name", corner_id)
        wear = c_data.get("remaining_wear_pct", 85.0)
        rate = c_data.get("wear_rate_pct_per_lap", 2.5)
        surf_t = c_data.get("surface_temp_c", 102.0)
        carc_t = c_data.get("carcass_temp_c", 98.0)
        t_status = c_data.get("thermal_status", "Optimal Window")
        t_color = c_data.get("thermal_color", "#00d2be")
        cliff = c_data.get("cliff_lap", 24)
        is_lim = c_data.get("is_limiting", False)
        lf = c_data.get("load_factor", 1.0)
        bar_col = get_bar_color(wear)
        cliff_color = "#ff1801" if is_lim else "#ffffff"

        if is_lim:
            lim_badge = (
                '<div style="background: rgba(255, 24, 1, 0.2); border: 1px solid #ff1801; color: #ff1801; '
                'font-size: 11px; font-weight: 800; border-radius: 4px; padding: 2px 6px; '
                'text-align: center; margin-bottom: 6px; letter-spacing: 0.5px;">'
                'CRITICAL LIMITING TYRE'
                '</div>'
            )
            border_style = "border: 2px solid #ff1801; box-shadow: 0 0 16px rgba(255, 24, 1, 0.45);"
        else:
            lim_badge = (
                f'<div style="font-size: 11px; color: #8c9ba5; margin-bottom: 6px; text-align: right;">'
                f'Circuit Load Factor: <b style="color: #ffffff;">{lf:.2f}x</b>'
                f'</div>'
            )
            border_style = "border: 1px solid rgba(255, 255, 255, 0.12);"

        return (
            f'<div style="background: rgba(18, 22, 32, 0.95); {border_style} border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
            f'<div style="font-family: Orbitron, sans-serif; font-size: 16px; font-weight: 800; color: #ffffff;">'
            f'{corner_id} <span style="font-size: 12px; color: #8c9ba5; font-weight: 400;">({name})</span>'
            f'</div>'
            f'<div style="color: {bar_col}; font-family: Orbitron, sans-serif; font-size: 20px; font-weight: 700;">'
            f'{wear:.1f}%'
            f'</div>'
            f'</div>'
            f'{lim_badge}'
            f'<div style="background: rgba(255, 255, 255, 0.08); border-radius: 4px; height: 7px; width: 100%; margin: 6px 0 10px 0; overflow: hidden;">'
            f'<div style="background: {bar_col}; width: {wear}%; height: 100%; border-radius: 4px;"></div>'
            f'</div>'
            f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; color: #8c9ba5; margin-top: 6px;">'
            f'<div>Wear / Lap: <b style="color: #ffffff;">-{rate:.1f}%</b></div>'
            f'<div>Est. Cliff: <b style="color: {cliff_color};">Lap {cliff}</b></div>'
            f'<div>Surf Temp: <b style="color: {t_color};">{surf_t:.1f}°C</b></div>'
            f'<div>Core Temp: <b style="color: #ffffff;">{carc_t:.1f}°C</b></div>'
            f'</div>'
            f'<div style="margin-top: 8px; font-size: 11px; color: {t_color}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">'
            f'● {t_status}'
            f'</div>'
            f'</div>'
        )

    # Top Header
    header_html = (
        f'<div style="background: rgba(14, 18, 27, 0.95); border: 1px solid rgba(0, 210, 190, 0.3); border-radius: 10px; padding: 12px 18px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
        f'<div>'
        f'<div style="font-family: Orbitron, sans-serif; font-size: 13px; color: #00d2be; letter-spacing: 1px; font-weight: 700;">FOUR-CORNER TYRE TELEMETRY</div>'
        f'<div style="color: #ffffff; font-size: 16px; font-weight: 700; margin-top: 2px;">{circuit_name} // <span style="color: #ffd700;">{limitation_text}</span></div>'
        f'<div style="color: #8c9ba5; font-size: 12px; margin-top: 2px;">{desc}</div>'
        f'</div>'
        f'<div style="background: rgba(255, 24, 1, 0.15); border: 1px solid #ff1801; border-radius: 8px; padding: 6px 14px; text-align: right;">'
        f'<div style="color: #8c9ba5; font-size: 10px; font-weight: 700; text-transform: uppercase;">Critical Limiting Corner</div>'
        f'<div style="font-family: Orbitron, sans-serif; font-size: 18px; font-weight: 800; color: #ff1801;">{limiting} ({tyres.get(limiting, {}).get("name", "Tyre")})</div>'
        f'<div style="color: #ffffff; font-size: 11px; font-weight: 600;">Pit Window Dictated: Lap {pit_lap}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Row 1: Front Axle (FL, FR)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(corner_card_html(fl, "FL"), unsafe_allow_html=True)
    with col_f2:
        st.markdown(corner_card_html(fr, "FR"), unsafe_allow_html=True)

    # Row 2: Rear Axle (RL, RR)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(corner_card_html(rl, "RL"), unsafe_allow_html=True)
    with col_r2:
        st.markdown(corner_card_html(rr, "RR"), unsafe_allow_html=True)

    # Clear spacing divider between rear wheels and whatever is rendered below
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
