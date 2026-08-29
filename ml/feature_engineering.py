import numpy as np
import pandas as pd
from typing import Dict, Any, List
from backend.config import settings


class FeatureEngineer:
    """Extracts and normalizes motorsport physics features for noise isolation."""

    @classmethod
    def transform(cls, laps_df: pd.DataFrame, session_info: Dict[str, Any]) -> pd.DataFrame:
        if laps_df.empty:
            return laps_df

        df = laps_df.copy()
        total_session_laps = session_info.get("total_laps", 57)
        track_temp = session_info.get("track_temp_c", 35.0)
        air_temp = session_info.get("air_temp_c", 25.0)

        # 1. Fuel Mass & Fuel Correction Feature
        # Race starts ~105kg; FP session starts ~45-50kg per stint
        if "fuel_kg" not in df.columns or df["fuel_kg"].isnull().all():
            df["fuel_kg"] = df["lap_number"].apply(
                lambda lap: max(5.0, 50.0 - (lap % 20) * settings.FUEL_BURN_KG_PER_LAP)
            )
        
        # Fuel penalty in seconds: 10kg ~ 0.30s
        df["fuel_penalty_sec"] = (df["fuel_kg"] / 10.0) * settings.FUEL_EFFECT_SEC_PER_10KG

        # 2. Track Evolution Index (t / T_total)
        # Rubbering-in curve: logarithmic gain over session
        if "track_evolution_index" not in df.columns or df["track_evolution_index"].isnull().all():
            df["track_evolution_index"] = (df["lap_number"] / float(total_session_laps)).clip(0.0, 1.0)
        
        df["track_evolution_delta_sec"] = -0.45 * np.log1p(df["track_evolution_index"] * 1.5)

        # 3. Driver Push Index (Aggression score 0.0 - 1.0)
        if "driver_push_index" not in df.columns or df["driver_push_index"].isnull().all():
            np.random.seed(42)
            df["driver_push_index"] = np.random.uniform(0.82, 0.95, size=len(df))
        
        df["driver_push_delta_sec"] = (1.0 - df["driver_push_index"]) * 0.35

        # 4. Traffic Penalty Detection
        if "traffic_flag" not in df.columns:
            df["traffic_flag"] = False
        
        df["traffic_penalty_sec"] = np.where(df["traffic_flag"], 0.65, 0.0)

        # 5. Thermal Features & Interactions
        # Optimal compound operating windows (C)
        optimal_temps = {"SOFT": 30.0, "MEDIUM": 36.0, "HARD": 42.0, "INTERMEDIATE": 20.0, "WET": 15.0}
        df["optimal_track_temp"] = df["compound"].map(optimal_temps).fillna(35.0)
        df["thermal_delta_c"] = track_temp - df["optimal_track_temp"]
        df["thermal_penalty_sec"] = (df["thermal_delta_c"] / 15.0) ** 2 * 0.08

        # Total expected external noise
        df["engineered_noise_sec"] = (
            df["fuel_penalty_sec"] +
            df["track_evolution_delta_sec"] +
            df["driver_push_delta_sec"] +
            df["traffic_penalty_sec"] +
            df["thermal_penalty_sec"]
        )

        return df

    @classmethod
    def get_feature_columns(cls) -> List[str]:
        return [
            "fuel_kg",
            "track_evolution_index",
            "driver_push_index",
            "traffic_penalty_sec",
            "thermal_delta_c",
        ]
