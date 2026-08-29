from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from backend.database.connection import get_db
from backend.services.data_service import DataService
from backend.services.ml_service import ml_service
from backend.schemas.prediction_schemas import PredictRequest
from ml.validation import ValidationService

router = APIRouter(prefix="/validate", tags=["Validation"])


@router.post("")
def validate_stint_endpoint(req: PredictRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        parts = req.session_id.split("_")
        season = int(parts[0]) if len(parts) > 0 else 2024
        gp = parts[1] if len(parts) > 1 else "Bahrain"
        sess_name = parts[2] if len(parts) > 2 else "FP2"

        session_info, laps_df, _ = DataService.load_and_sync_session(db, season, gp, sess_name)
        pred_res = ml_service.run_prediction(db, req, laps_df, session_info)
        
        curve_dicts = [p.dict() for p in pred_res.curve]
        comp_laps = laps_df[
            (laps_df["driver_code"] == req.driver_code) & 
            (laps_df["compound"] == req.compound.upper())
        ].to_dict(orient="records")

        eval_result = ValidationService.evaluate(curve_dicts, comp_laps)
        return {
            "driver_code": req.driver_code,
            "compound": req.compound,
            "metrics": eval_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
