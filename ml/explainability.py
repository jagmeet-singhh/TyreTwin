import numpy as np
import pandas as pd
from typing import List, Dict, Any


class ExplainabilityEngine:
    """Calculates SHAP-style attribution for lap time external noise breakdown."""

    @classmethod
    def explain_lap(cls, lap_row: pd.Series) -> List[Dict[str, Any]]:
        fuel = float(lap_row.get("fuel_penalty_sec", 0.30))
        traffic = float(lap_row.get("traffic_penalty_sec", 0.0))
        track_evo = float(lap_row.get("track_evolution_delta_sec", -0.15))
        driver_push = float(lap_row.get("driver_push_delta_sec", 0.05))
        thermal = float(lap_row.get("thermal_penalty_sec", 0.02))

        return [
            {
                "feature": "fuel_mass",
                "display_name": "Fuel Load",
                "delta_sec": round(fuel, 3),
                "description": f"+{round(fuel, 2)}s due to {round(lap_row.get('fuel_kg', 40), 1)}kg onboard fuel weight",
            },
            {
                "feature": "traffic",
                "display_name": "Traffic & Dirty Air",
                "delta_sec": round(traffic, 3),
                "description": "+0.65s traffic congestion / wake disturbance" if traffic > 0 else "0.00s clean air",
            },
            {
                "feature": "track_evolution",
                "display_name": "Track Grip Evolution",
                "delta_sec": round(track_evo, 3),
                "description": f"{round(track_evo, 2)}s grip improvement from rubbered-in racing line",
            },
            {
                "feature": "driver_push",
                "display_name": "Driver Push Aggression",
                "delta_sec": round(driver_push, 3),
                "description": f"+{round(driver_push, 2)}s pace conservation vs 100% qualy push",
            },
            {
                "feature": "thermal",
                "display_name": "Track Surface Temp",
                "delta_sec": round(thermal, 3),
                "description": f"+{round(thermal, 2)}s compound thermal delta",
            },
        ]
