from typing import Optional, List
from pydantic import BaseModel


class SessionLoadRequest(BaseModel):
    season: int = 2024
    grand_prix: str = "Bahrain"
    session_name: str = "FP2"  # FP1, FP2, FP3, Q, R
    force_refresh: bool = False


class SessionInfoResponse(BaseModel):
    id: str
    season: int
    grand_prix: str
    session_name: str
    track_name: str
    air_temp_c: float
    track_temp_c: float
    total_laps: int
    drivers_count: int
    laps_count: int


class DriverInfoResponse(BaseModel):
    code: str
    number: int
    full_name: str
    team_name: str
    team_color: str
