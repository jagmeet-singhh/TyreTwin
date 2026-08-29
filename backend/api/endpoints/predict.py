from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.schemas.prediction_schemas import PredictRequest, PredictResponse
from backend.services.ml_service import ml_service
from backend.services.data_service import DataService

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictResponse)
def predict_degradation_endpoint(req: PredictRequest, db: Session = Depends(get_db)):
    try:
        # Parse session id e.g. 2024_Bahrain_FP2
        parts = req.session_id.split("_")
        season = int(parts[0]) if len(parts) > 0 else 2024
        gp = parts[1] if len(parts) > 1 else "Bahrain"
        sess_name = parts[2] if len(parts) > 2 else "FP2"

        session_info, laps_df, _ = DataService.load_and_sync_session(db, season, gp, sess_name)
        res = ml_service.run_prediction(db, req, laps_df, session_info)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
