import numpy as np
from typing import Dict, Any, List
from backend.schemas.strategy_schemas import (
    StrategyRequest, StrategyResponse, StrategyRecommendation, StintPlan
)
from backend.config import settings


class StrategyService:
    """Calculates optimal F1 pit stop windows, compound crossover, and race pace simulations."""

    @classmethod
    def calculate_strategy(cls, req: StrategyRequest, base_pace: float = 91.5) -> StrategyResponse:
        target_laps = req.target_race_laps
        current_lap = req.current_lap
        current_compound = req.current_compound.upper()
        pit_loss = settings.PIT_LOSS_SEC  # ~22.0 sec pit lane time loss

        # Compound speed & degradation characteristics
        compound_profiles = {
            "SOFT": {"pace_offset": -0.85, "deg_rate": 0.088, "max_stint": 18, "cliff": 16},
            "MEDIUM": {"pace_offset": 0.00, "deg_rate": 0.054, "max_stint": 28, "cliff": 26},
            "HARD": {"pace_offset": 0.75, "deg_rate": 0.032, "max_stint": 40, "cliff": 36},
        }

        curr_prof = compound_profiles.get(current_compound, compound_profiles["MEDIUM"])
        curr_cliff = curr_prof["cliff"]
        tyre_health_pct = max(0.0, 100.0 - (req.tyre_age / float(curr_cliff)) * 80.0)

        # 1. Generate 1-Stop Strategy: (e.g. Current -> HARD/MEDIUM)
        # Optimal 1-stop pit lap
        opt_pit_lap_1 = min(target_laps - 15, max(current_lap, curr_cliff - 2))
        stint1_len = opt_pit_lap_1
        stint2_len = target_laps - opt_pit_lap_1
        stint2_comp = "HARD" if current_compound in ["SOFT", "MEDIUM"] else "MEDIUM"
        prof2 = compound_profiles[stint2_comp]

        # Calculate time for Stint 1
        stint1_pace = base_pace + curr_prof["pace_offset"] + (stint1_len * curr_prof["deg_rate"] * 0.5)
        stint1_total = (stint1_pace * stint1_len)

        # Calculate time for Stint 2
        stint2_pace = base_pace + prof2["pace_offset"] + (stint2_len * prof2["deg_rate"] * 0.5)
        stint2_total = (stint2_pace * stint2_len) + pit_loss

        total_1stop_time = stint1_total + stint2_total

        strat_1stop = StrategyRecommendation(
            strategy_name=f"1-Stop Plan A ({current_compound} -> {stint2_comp})",
            stops=1,
            stints=[
                StintPlan(
                    stint_number=1,
                    compound=current_compound,
                    start_lap=1,
                    end_lap=opt_pit_lap_1,
                    stint_length=stint1_len,
                    avg_pace_sec=round(stint1_pace, 3),
                    total_stint_time_sec=round(stint1_total, 2),
                ),
                StintPlan(
                    stint_number=2,
                    compound=stint2_comp,
                    start_lap=opt_pit_lap_1 + 1,
                    end_lap=target_laps,
                    stint_length=stint2_len,
                    avg_pace_sec=round(stint2_pace, 3),
                    total_stint_time_sec=round(stint2_total, 2),
                ),
            ],
            total_race_time_sec=round(total_1stop_time, 2),
            pit_laps=[opt_pit_lap_1],
            pit_window=f"Laps {max(1, opt_pit_lap_1 - 2)} - {opt_pit_lap_1 + 2}",
            tyre_cliff_warning_lap=curr_cliff,
            recommendation_summary=(
                f"Optimal pit stop at Lap {opt_pit_lap_1} for {stint2_comp}. "
                f"Tyre cliff arrives around Lap {curr_cliff}. Crossover pace delta favours fresh {stint2_comp}."
            ),
            crossover_lap=opt_pit_lap_1,
            is_optimal=True,
        )

        # 2. Generate 2-Stop Strategy: (e.g. Current -> MEDIUM -> SOFT)
        pit1_2stop = max(current_lap, int(target_laps * 0.32))
        pit2_2stop = int(target_laps * 0.68)
        s1_2 = pit1_2stop
        s2_2 = pit2_2stop - pit1_2stop
        s3_2 = target_laps - pit2_2stop

        total_2stop_time = (
            (base_pace + curr_prof["pace_offset"] + s1_2 * curr_prof["deg_rate"] * 0.5) * s1_2 +
            (base_pace + compound_profiles["MEDIUM"]["pace_offset"] + s2_2 * compound_profiles["MEDIUM"]["deg_rate"] * 0.5) * s2_2 +
            (base_pace + compound_profiles["SOFT"]["pace_offset"] + s3_2 * compound_profiles["SOFT"]["deg_rate"] * 0.5) * s3_2 +
            (pit_loss * 2)
        )

        strat_2stop = StrategyRecommendation(
            strategy_name=f"2-Stop Plan B ({current_compound} -> MEDIUM -> SOFT)",
            stops=2,
            stints=[
                StintPlan(
                    stint_number=1,
                    compound=current_compound,
                    start_lap=1,
                    end_lap=pit1_2stop,
                    stint_length=s1_2,
                    avg_pace_sec=round(base_pace + curr_prof["pace_offset"] + s1_2 * curr_prof["deg_rate"] * 0.5, 3),
                    total_stint_time_sec=round(s1_2 * base_pace, 2),
                ),
                StintPlan(
                    stint_number=2,
                    compound="MEDIUM",
                    start_lap=pit1_2stop + 1,
                    end_lap=pit2_2stop,
                    stint_length=s2_2,
                    avg_pace_sec=round(base_pace + compound_profiles["MEDIUM"]["pace_offset"] + s2_2 * 0.054 * 0.5, 3),
                    total_stint_time_sec=round(s2_2 * base_pace + pit_loss, 2),
                ),
                StintPlan(
                    stint_number=3,
                    compound="SOFT",
                    start_lap=pit2_2stop + 1,
                    end_lap=target_laps,
                    stint_length=s3_2,
                    avg_pace_sec=round(base_pace + compound_profiles["SOFT"]["pace_offset"] + s3_2 * 0.088 * 0.5, 3),
                    total_stint_time_sec=round(s3_2 * base_pace + pit_loss, 2),
                ),
            ],
            total_race_time_sec=round(total_2stop_time, 2),
            pit_laps=[pit1_2stop, pit2_2stop],
            pit_window=f"Pit 1: Laps {pit1_2stop-1}-{pit1_2stop+1} | Pit 2: Laps {pit2_2stop-1}-{pit2_2stop+1}",
            tyre_cliff_warning_lap=curr_cliff,
            recommendation_summary=(
                f"Aggressive 2-stop undercut strategy. Pit Laps {pit1_2stop} & {pit2_2stop}. "
                f"Provides +0.45s/lap tyre grip advantage in final stint."
            ),
            crossover_lap=pit1_2stop,
            is_optimal=False,
        )

        # Pace comparison curve across race laps
        pace_curve = []
        for l in range(1, target_laps + 1):
            # 1-stop pace
            if l <= opt_pit_lap_1:
                p1 = base_pace + curr_prof["pace_offset"] + (l * curr_prof["deg_rate"])
            else:
                l_stint = l - opt_pit_lap_1
                p1 = base_pace + prof2["pace_offset"] + (l_stint * prof2["deg_rate"])

            # 2-stop pace
            if l <= pit1_2stop:
                p2 = base_pace + curr_prof["pace_offset"] + (l * curr_prof["deg_rate"])
            elif l <= pit2_2stop:
                l_stint = l - pit1_2stop
                p2 = base_pace + compound_profiles["MEDIUM"]["pace_offset"] + (l_stint * 0.054)
            else:
                l_stint = l - pit2_2stop
                p2 = base_pace + compound_profiles["SOFT"]["pace_offset"] + (l_stint * 0.088)

            pace_curve.append({
                "lap": l,
                "strategy_1stop_pace_sec": round(p1, 3),
                "strategy_2stop_pace_sec": round(p2, 3),
            })

        return StrategyResponse(
            driver_code=req.driver_code,
            current_lap=current_lap,
            current_compound=current_compound,
            tyre_health_pct=round(tyre_health_pct, 1),
            recommended_strategy=strat_1stop,
            alternative_strategies=[strat_2stop],
            pace_comparison_curve=pace_curve,
        )
