from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    session_id: str
    driver_code: str
    compound: str
    stint_length: int = 25
    ambient_temp_delta: float = 0.0  # Scenario adjustment


class DegradationPoint(BaseModel):
    lap: int
    tyre_age: int
    predicted_lap_time_sec: float
    lower_bound_sec: float  # -2 sigma
    upper_bound_sec: float  # +2 sigma
    uncertainty_sigma: float
    wear_index_pct: float   # 0 to 100%


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    delta_sec: float
    description: str


class PredictResponse(BaseModel):
    session_id: str
    driver_code: str
    compound: str
    base_clean_lap_sec: float
    degradation_slope_sec_per_lap: float
    cliff_lap: int
    remaining_life_laps: int
    recommended_pit_lap: int
    pit_window_start: int
    pit_window_end: int
    confidence_pct: float
    curve: List[DegradationPoint]
    observed_laps: List[Dict[str, Any]]
    clean_laps: List[Dict[str, Any]]
    feature_attributions: List[FeatureContribution]
