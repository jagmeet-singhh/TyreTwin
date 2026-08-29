import pytest
from backend.services.strategy_service import StrategyService
from backend.schemas.strategy_schemas import StrategyRequest

def test_strategy_crossover_optimization():
    req = StrategyRequest(
        session_id="2024_Bahrain_FP2",
        driver_code="VER",
        current_lap=10,
        current_compound="SOFT",
        tyre_age=10,
        target_race_laps=57
    )
    res = StrategyService.calculate_strategy(req, base_pace=91.5)
    assert res.recommended_strategy.stops == 1
    assert res.recommended_strategy.is_optimal == True
    assert len(res.alternative_strategies) >= 1
    assert res.recommended_strategy.pit_laps[0] > 10
