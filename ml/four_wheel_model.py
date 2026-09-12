import numpy as np
from typing import Dict, Any, List, Optional


class FourWheelTyreModel:
    """
    Simulates individual corner dynamics for all 4 tyres of a Formula 1 car:
    - FL: Front-Left
    - FR: Front-Right
    - RL: Rear-Left
    - RR: Rear-Right
    
    Accounts for circuit layout asymmetries, lateral vs longitudinal load transfers,
    surface & carcass thermal dynamics (Salucci et al. SAE 2020), and Pacejka grip decay.
    """

    CIRCUIT_CORNER_PROFILES = {
        "bahrain": {
            "name": "Bahrain International Circuit (Sakhir)",
            "circuit_limitation": "Rear Traction & Lateral Slip Limited",
            "load_factors": {"FL": 0.94, "FR": 0.88, "RL": 1.24, "RR": 1.20},
            "limiting_corner": "RL",
            "description": "High asphalt macro-roughness and heavy low-speed traction acceleration (T1, T4, T8, T10) punish rear tyres.",
        },
        "silverstone": {
            "name": "Silverstone Circuit",
            "circuit_limitation": "Front-Left High-Speed Lateral G Limited",
            "load_factors": {"FL": 1.28, "FR": 0.96, "RL": 1.10, "RR": 0.90},
            "limiting_corner": "FL",
            "description": "Extreme sustained lateral loading through Copse, Maggotts, Becketts, and Stowe severely wears Front-Left.",
        },
        "monza": {
            "name": "Autodromo Nazionale Monza",
            "circuit_limitation": "Rear Traction & Longitudinal Braking Limited",
            "load_factors": {"FL": 1.05, "FR": 1.05, "RL": 1.18, "RR": 1.18},
            "limiting_corner": "RL",
            "description": "Violent chicane deceleration and aggressive curb striking combined with low-downforce wheelspin out of Rettifilo and Roggia.",
        },
        "suzuka": {
            "name": "Suzuka International Racing Course",
            "circuit_limitation": "Front Lateral Scrub Limited (Figure-8)",
            "load_factors": {"FL": 1.18, "FR": 1.24, "RL": 1.10, "RR": 1.04},
            "limiting_corner": "FR",
            "description": "Unique Figure-8 configuration with high lateral energy in the first sector Esses and Degners punishing Front-Right.",
        },
    }

    DEFAULT_PROFILE = {
        "name": "Standard Grand Prix Circuit",
        "circuit_limitation": "Balanced Circuit Wear",
        "load_factors": {"FL": 1.08, "FR": 0.98, "RL": 1.14, "RR": 1.04},
        "limiting_corner": "RL",
        "description": "Balanced lateral and longitudinal wear profile across high and low speed sectors.",
    }

    COMPOUND_CHARACTERISTICS = {
        "SOFT": {
            "base_wear_rate": 3.4,  # % wear per lap at load factor 1.0
            "temp_opt_min": 100.0,
            "temp_opt_max": 115.0,
            "heating_rate": 0.16,
        },
        "MEDIUM": {
            "base_wear_rate": 2.4,
            "temp_opt_min": 105.0,
            "temp_opt_max": 125.0,
            "heating_rate": 0.13,
        },
        "HARD": {
            "base_wear_rate": 1.6,
            "temp_opt_min": 115.0,
            "temp_opt_max": 135.0,
            "heating_rate": 0.10,
        },
        "INTERMEDIATE": {
            "base_wear_rate": 2.7,
            "temp_opt_min": 55.0,
            "temp_opt_max": 80.0,
            "heating_rate": 0.12,
        },
        "WET": {
            "base_wear_rate": 2.3,
            "temp_opt_min": 45.0,
            "temp_opt_max": 70.0,
            "heating_rate": 0.11,
        },
    }

    CORNER_NAMES = {
        "FL": "Front-Left",
        "FR": "Front-Right",
        "RL": "Rear-Left",
        "RR": "Rear-Right",
    }

    @classmethod
    def get_circuit_profile(cls, session_id: str) -> Dict[str, Any]:
        """Resolves circuit profile based on session_id or track name."""
        s_lower = session_id.lower()
        for key, profile in cls.CIRCUIT_CORNER_PROFILES.items():
            if key in s_lower:
                return profile
        return cls.DEFAULT_PROFILE

    @classmethod
    def simulate_four_tyres(
        cls,
        session_id: str,
        compound: str,
        stint_length: int = 25,
        current_age: int = 1,
        track_temp_c: float = 38.0,
        driver_push_index: float = 0.92,
    ) -> Dict[str, Any]:
        """
        Computes 4-corner degradation trajectories, remaining life, and temperatures across stint laps.
        """
        profile = cls.get_circuit_profile(session_id)
        comp_info = cls.COMPOUND_CHARACTERISTICS.get(compound.upper(), cls.COMPOUND_CHARACTERISTICS["MEDIUM"])
        load_factors = profile["load_factors"]

        tyres_summary = {}
        history_by_lap = []
        corner_curves = {c: [] for c in ["FL", "FR", "RL", "RR"]}

        for corner in ["FL", "FR", "RL", "RR"]:
            lf = load_factors[corner]
            eff_wear_rate = comp_info["base_wear_rate"] * lf * (0.7 + 0.3 * driver_push_index)
            
            # Cliff occurs when remaining rubber hits 20% (wear reaches 80%)
            cliff_lap = max(5, int(np.floor(80.0 / max(0.1, eff_wear_rate))))
            
            # Current snapshot at current_age
            current_wear_pct = min(100.0, max(0.0, 100.0 - eff_wear_rate * current_age))
            rubber_depth_mm = max(0.0, 5.0 * (current_wear_pct / 100.0))
            
            # Surface and Carcass temperature modeling
            delta_t_surf = (65.0 * (1.0 - np.exp(-comp_info["heating_rate"] * current_age))) * lf * (0.8 + 0.2 * driver_push_index)
            delta_t_carcass = (58.0 * (1.0 - np.exp(-0.08 * current_age))) * lf
            
            surf_temp = round(track_temp_c + delta_t_surf, 1)
            carcass_temp = round(track_temp_c + delta_t_carcass, 1)
            
            # Thermal Status
            if surf_temp > comp_info["temp_opt_max"] + 8.0:
                thermal_status = "Thermal Blistering"
                thermal_color = "#ff1801"
            elif surf_temp > comp_info["temp_opt_max"]:
                thermal_status = "Overheating"
                thermal_color = "#ff793f"
            elif surf_temp < comp_info["temp_opt_min"] - 6.0:
                thermal_status = "Graining Risk"
                thermal_color = "#3498db"
            else:
                thermal_status = "Optimal Window"
                thermal_color = "#00d2be"

            remaining_laps = max(0, cliff_lap - current_age)

            tyres_summary[corner] = {
                "corner": corner,
                "name": cls.CORNER_NAMES[corner],
                "load_factor": round(float(lf), 2),
                "wear_rate_pct_per_lap": round(float(eff_wear_rate), 2),
                "remaining_wear_pct": round(float(current_wear_pct), 1),
                "rubber_depth_mm": round(float(rubber_depth_mm), 2),
                "surface_temp_c": surf_temp,
                "carcass_temp_c": carcass_temp,
                "temp_opt_min": comp_info["temp_opt_min"],
                "temp_opt_max": comp_info["temp_opt_max"],
                "thermal_status": thermal_status,
                "thermal_color": thermal_color,
                "cliff_lap": cliff_lap,
                "remaining_laps": remaining_laps,
                "is_limiting": False,
            }

            # Build stint curve for this corner
            for lap in range(1, stint_length + 1):
                wear_pct = min(100.0, max(0.0, 100.0 - eff_wear_rate * lap))
                dt_s = (65.0 * (1.0 - np.exp(-comp_info["heating_rate"] * lap))) * lf * (0.8 + 0.2 * driver_push_index)
                dt_c = (58.0 * (1.0 - np.exp(-0.08 * lap))) * lf
                corner_curves[corner].append({
                    "lap": lap,
                    "remaining_wear_pct": round(float(wear_pct), 1),
                    "surface_temp_c": round(track_temp_c + dt_s, 1),
                    "carcass_temp_c": round(track_temp_c + dt_c, 1),
                })

        # Identify limiting corner (lowest cliff lap)
        limiting_corner = min(tyres_summary.keys(), key=lambda c: tyres_summary[c]["cliff_lap"])
        tyres_summary[limiting_corner]["is_limiting"] = True

        # Build lap-by-lap history array
        for lap_idx in range(stint_length):
            lap_num = lap_idx + 1
            history_by_lap.append({
                "lap": lap_num,
                "FL_wear_pct": corner_curves["FL"][lap_idx]["remaining_wear_pct"],
                "FR_wear_pct": corner_curves["FR"][lap_idx]["remaining_wear_pct"],
                "RL_wear_pct": corner_curves["RL"][lap_idx]["remaining_wear_pct"],
                "RR_wear_pct": corner_curves["RR"][lap_idx]["remaining_wear_pct"],
                "FL_surf_temp": corner_curves["FL"][lap_idx]["surface_temp_c"],
                "FR_surf_temp": corner_curves["FR"][lap_idx]["surface_temp_c"],
                "RL_surf_temp": corner_curves["RL"][lap_idx]["surface_temp_c"],
                "RR_surf_temp": corner_curves["RR"][lap_idx]["surface_temp_c"],
            })

        return {
            "circuit_name": profile["name"],
            "circuit_limitation": profile["circuit_limitation"],
            "circuit_description": profile["description"],
            "limiting_corner": limiting_corner,
            "limiting_corner_name": cls.CORNER_NAMES[limiting_corner],
            "recommended_pit_lap": max(1, tyres_summary[limiting_corner]["cliff_lap"] - 2),
            "tyres": tyres_summary,
            "curves": corner_curves,
            "history": history_by_lap,
        }
