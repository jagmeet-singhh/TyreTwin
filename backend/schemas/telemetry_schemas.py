from typing import List, Optional
from pydantic import BaseModel


class TelemetryPoint(BaseModel):
    distance_m: float
    speed_kmh: float
    throttle_pct: float
    brake_pct: float
    rpm: int
    gear: int
    drs: int
    time_sec: float


class LapTelemetryResponse(BaseModel):
    session_id: str
    driver_code: str
    lap_number: int
    lap_time_sec: float
    compound: str
    clean_lap_time_sec: float
    noise_sec: float
    trace: List[TelemetryPoint]
