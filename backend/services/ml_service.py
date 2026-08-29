import logging
import pandas as pd
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.database import crud
from ml.train_noise import NoiseIsolationModel
from ml.train_degradation import TyreDegradationModel
from ml.predict import PredictionService
from backend.schemas.prediction_schemas import PredictRequest, PredictResponse

logger = logging.getLogger("TyreIQ.MLService")


class MLService:
    """Singleton ML Service orchestrating models and DB cache."""

    def __init__(self):
        self.noise_model = NoiseIsolationModel.load()
        self.deg_model = TyreDegradationModel.load()
        self.pred_service = PredictionService(self.noise_model, self.deg_model)

    def run_prediction(
        self, db: Session, req: PredictRequest, laps_df: pd.DataFrame, session_info: Dict[str, Any]
    ) -> PredictResponse:
        # Check driver laps
        driver_laps = laps_df[laps_df["driver_code"] == req.driver_code] if not laps_df.empty else pd.DataFrame()
        
        # Execute prediction
        pred_res = self.pred_service.predict_degradation(
            session_id=req.session_id,
            driver_code=req.driver_code,
            compound=req.compound,
            driver_laps_df=driver_laps,
            session_info=session_info,
            stint_length=req.stint_length,
        )

        return pred_res


ml_service = MLService()
