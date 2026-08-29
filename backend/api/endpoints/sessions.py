from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from backend.database.connection import get_db
from backend.database import crud
from backend.schemas.session_schemas import SessionLoadRequest, SessionInfoResponse, DriverInfoResponse
from backend.services.data_service import DataService
from ml.fastf1_loader import DRIVERS_METADATA

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("/load", response_model=SessionInfoResponse)
def load_session_endpoint(req: SessionLoadRequest, db: Session = Depends(get_db)):
    try:
        session_info, laps_df, _ = DataService.load_and_sync_session(
            db, req.season, req.grand_prix, req.session_name, force_refresh=req.force_refresh
        )
        return SessionInfoResponse(
            id=session_info["id"],
            season=session_info["season"],
            grand_prix=session_info["grand_prix"],
            session_name=session_info["session_name"],
            track_name=session_info["track_name"],
            air_temp_c=session_info["air_temp_c"],
            track_temp_c=session_info["track_temp_c"],
            total_laps=session_info["total_laps"],
            drivers_count=len(DRIVERS_METADATA),
            laps_count=len(laps_df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")


@router.get("/list", response_model=List[SessionInfoResponse])
def list_sessions_endpoint(season: Optional[int] = None, db: Session = Depends(get_db)):
    sessions = crud.list_sessions(db, season)
    return [
        SessionInfoResponse(
            id=s.id,
            season=s.season,
            grand_prix=s.grand_prix,
            session_name=s.session_name,
            track_name=s.track_name,
            air_temp_c=s.air_temp_c,
            track_temp_c=s.track_temp_c,
            total_laps=s.total_laps,
            drivers_count=len(DRIVERS_METADATA),
            laps_count=len(s.laps),
        )
        for s in sessions
    ]


@router.get("/drivers", response_model=List[DriverInfoResponse])
def list_drivers_endpoint(db: Session = Depends(get_db)):
    drivers = crud.list_drivers(db)
    if not drivers:
        return [DriverInfoResponse(**d) for d in DRIVERS_METADATA]
    return [
        DriverInfoResponse(
            code=d.code,
            number=d.number,
            full_name=d.full_name,
            team_name=d.team_name,
            team_color=d.team_color,
        )
        for d in drivers
    ]
