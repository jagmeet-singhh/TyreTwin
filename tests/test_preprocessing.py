import pytest
import pandas as pd
from ml.preprocessing import DataPreprocessor

def test_clean_laps_filters_in_out_laps():
    df = pd.DataFrame([
        {"driver_code": "VER", "compound": "MEDIUM", "lap_number": 1, "lap_time_sec": 110.0, "is_valid": False, "is_pit_out": True, "is_pit_in": False, "track_status": "1"},
        {"driver_code": "VER", "compound": "MEDIUM", "lap_number": 2, "lap_time_sec": 91.5, "is_valid": True, "is_pit_out": False, "is_pit_in": False, "track_status": "1"},
        {"driver_code": "VER", "compound": "MEDIUM", "lap_number": 3, "lap_time_sec": 91.8, "is_valid": True, "is_pit_out": False, "is_pit_in": False, "track_status": "1"},
        {"driver_code": "VER", "compound": "MEDIUM", "lap_number": 4, "lap_time_sec": 105.0, "is_valid": False, "is_pit_out": False, "is_pit_in": True, "track_status": "1"},
    ])
    cleaned = DataPreprocessor.clean_laps_data(df)
    assert len(cleaned) == 2
    assert set(cleaned["lap_number"]) == {2, 3}
