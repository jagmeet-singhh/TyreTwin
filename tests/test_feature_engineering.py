import pytest
import pandas as pd
from ml.feature_engineering import FeatureEngineer

def test_feature_engineering_fuel_penalty():
    df = pd.DataFrame([
        {"lap_number": 1, "compound": "SOFT", "fuel_kg": 40.0, "track_evolution_index": 0.1, "traffic_flag": False},
        {"lap_number": 10, "compound": "SOFT", "fuel_kg": 20.0, "track_evolution_index": 0.5, "traffic_flag": True},
    ])
    session_info = {"total_laps": 57, "track_temp_c": 35.0, "air_temp_c": 25.0}
    res = FeatureEngineer.transform(df, session_info)
    assert "fuel_penalty_sec" in res.columns
    assert "engineered_noise_sec" in res.columns
    # 40kg should produce higher fuel penalty than 20kg
    assert res.loc[0, "fuel_penalty_sec"] > res.loc[1, "fuel_penalty_sec"]
    # Lap 2 has traffic flag -> traffic penalty > 0
    assert res.loc[1, "traffic_penalty_sec"] > 0.0
