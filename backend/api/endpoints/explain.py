from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import pandas as pd
from backend.schemas.prediction_schemas import FeatureContribution
from ml.explainability import ExplainabilityEngine

router = APIRouter(prefix="/explain", tags=["Explainability"])


@router.post("", response_model=List[FeatureContribution])
def explain_lap_endpoint(lap_data: Dict[str, Any]):
    try:
        series = pd.Series(lap_data)
        raw_attrs = ExplainabilityEngine.explain_lap(series)
        return [FeatureContribution(**a) for a in raw_attrs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attribution failed: {str(e)}")
