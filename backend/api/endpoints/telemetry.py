from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from backend.database.connection import get_db
from backend.services.data_service import DataService
from backend.schemas.telemetry_schemas import LapTelemetryResponse, TelemetryPoint

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.get("", response_model=LapTelemetryResponse)
def get_telemetry_endpoint(
    session_id: str = Query("2024_Bahrain_FP2"),
    driver_code: str = Query("VER"),
    lap_number: int = Query(5),
    db: Session = Depends(get_db)
):
    try:
        parts = session_id.split("_")
        season = int(parts[0]) if len(parts) > 0 else 2024
        gp = parts[1] if len(parts) > 1 else "Bahrain"
        sess_name = parts[2] if len(parts) > 2 else "FP2"

        session_info, laps_df, tel_dict = DataService.load_and_sync_session(db, season, gp, sess_name)
        
        tel_key = f"{driver_code}_{lap_number}"
        if tel_key in tel_dict:
            t_df = tel_dict[tel_key]
        else:
            # Generate on the fly for requested lap
            t_df = DataService.load_and_sync_session(db, season, gp, sess_name)[2].get(f"{driver_code}_3")
            if t_df is None or t_df.empty:
                from ml.fastf1_loader import FastF1DataLoader
                t_df = FastF1DataLoader.generate_benchmark_session(session_id)[2].get(f"{driver_code}_3")

        trace_points = [TelemetryPoint(**row) for _, row in t_df.iterrows()] if (t_df is not None and not t_df.empty) else []

        driver_laps = laps_df[(laps_df["driver_code"] == driver_code) & (laps_df["lap_number"] == lap_number)]
        lap_time = float(driver_laps.iloc[0]["lap_time_sec"]) if not driver_laps.empty else 91.5
        compound = str(driver_laps.iloc[0]["compound"]) if not driver_laps.empty else "MEDIUM"
        clean_time = float(driver_laps.iloc[0].get("clean_lap_time_sec", lap_time - 0.35)) if not driver_laps.empty else lap_time - 0.35
        noise = lap_time - clean_time

        return LapTelemetryResponse(
            session_id=session_id,
            driver_code=driver_code,
            lap_number=lap_number,
            lap_time_sec=round(lap_time, 3),
            compound=compound,
            clean_lap_time_sec=round(clean_time, 3),
            noise_sec=round(noise, 3),
            trace=trace_points,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch telemetry: {str(e)}")
