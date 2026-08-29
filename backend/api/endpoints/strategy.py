from fastapi import APIRouter, HTTPException
from backend.schemas.strategy_schemas import StrategyRequest, StrategyResponse
from backend.services.strategy_service import StrategyService
from ml.fastf1_loader import BENCHMARK_CONFIGS

router = APIRouter(prefix="/strategy", tags=["Strategy"])


@router.post("", response_model=StrategyResponse)
def recommend_strategy_endpoint(req: StrategyRequest):
    try:
        cfg = BENCHMARK_CONFIGS.get(req.session_id, BENCHMARK_CONFIGS["2024_Bahrain_FP2"])
        base_pace = cfg.get("base_lap_sec", 91.5)
        return StrategyService.calculate_strategy(req, base_pace=base_pace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy optimization error: {str(e)}")
