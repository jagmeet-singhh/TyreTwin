from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class StrategyRequest(BaseModel):
    session_id: str
    driver_code: str
    current_lap: int = 10
    current_compound: str = "SOFT"
    tyre_age: int = 10
    planned_stops: int = 1
    alternative_compounds: List[str] = ["MEDIUM", "HARD"]
    target_race_laps: int = 57


class StintPlan(BaseModel):
    stint_number: int
    compound: str
    start_lap: int
    end_lap: int
    stint_length: int
    avg_pace_sec: float
    total_stint_time_sec: float


class StrategyRecommendation(BaseModel):
    strategy_name: str
    stops: int
    stints: List[StintPlan]
    total_race_time_sec: float
    pit_laps: List[int]
    pit_window: str
    tyre_cliff_warning_lap: int
    recommendation_summary: str
    crossover_lap: int
    is_optimal: bool


class StrategyResponse(BaseModel):
    driver_code: str
    current_lap: int
    current_compound: str
    tyre_health_pct: float
    recommended_strategy: StrategyRecommendation
    alternative_strategies: List[StrategyRecommendation]
    pace_comparison_curve: List[Dict[str, Any]]
