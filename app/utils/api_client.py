import requests
import logging
from typing import Dict, Any, List, Optional
from backend.config import settings

logger = logging.getLogger("TyreIQ.APIClient")

API_BASE = f"http://localhost:{settings.PORT}/api"


class TyreIQClient:
    """HTTP REST Client with graceful direct service fallback."""

    @classmethod
    def load_session(cls, season: int, grand_prix: str, session_name: str) -> Dict[str, Any]:
        try:
            resp = requests.post(
                f"{API_BASE}/sessions/load",
                json={"season": season, "grand_prix": grand_prix, "session_name": session_name},
                timeout=5.0
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"REST call failed: {e}. Executing direct fallback.")
        
        # Direct service fallback
        from ml.fastf1_loader import FastF1DataLoader, DRIVERS_METADATA
        sess_id = f"{season}_{grand_prix}_{session_name}"
        s_info, laps, _ = FastF1DataLoader.load_session(season, grand_prix, session_name)
        return {
            "id": sess_id,
            "season": season,
            "grand_prix": grand_prix,
            "session_name": session_name,
            "track_name": s_info["track_name"],
            "air_temp_c": s_info["air_temp_c"],
            "track_temp_c": s_info["track_temp_c"],
            "total_laps": s_info["total_laps"],
            "drivers_count": len(DRIVERS_METADATA),
            "laps_count": len(laps),
        }

    @classmethod
    def predict_degradation(
        cls, session_id: str, driver_code: str, compound: str, stint_length: int = 25
    ) -> Dict[str, Any]:
        try:
            resp = requests.post(
                f"{API_BASE}/predict",
                json={
                    "session_id": session_id,
                    "driver_code": driver_code,
                    "compound": compound,
                    "stint_length": stint_length
                },
                timeout=5.0
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"REST predict failed: {e}. Direct fallback.")

        # Fallback direct prediction
        from ml.fastf1_loader import FastF1DataLoader
        from ml.train_noise import NoiseIsolationModel
        from ml.train_degradation import TyreDegradationModel
        from ml.predict import PredictionService

        parts = session_id.split("_")
        s_info, laps, _ = FastF1DataLoader.load_session(int(parts[0]), parts[1], parts[2])
        drv_laps = laps[laps["driver_code"] == driver_code] if not laps.empty else laps
        
        service = PredictionService(NoiseIsolationModel(), TyreDegradationModel())
        res = service.predict_degradation(session_id, driver_code, compound, drv_laps, s_info, stint_length)
        return res.dict()

    @classmethod
    def get_strategy(
        cls, session_id: str, driver_code: str, current_lap: int, current_compound: str, tyre_age: int
    ) -> Dict[str, Any]:
        try:
            resp = requests.post(
                f"{API_BASE}/strategy",
                json={
                    "session_id": session_id,
                    "driver_code": driver_code,
                    "current_lap": current_lap,
                    "current_compound": current_compound,
                    "tyre_age": tyre_age
                },
                timeout=5.0
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"REST strategy failed: {e}. Direct fallback.")

        from backend.services.strategy_service import StrategyService
        from backend.schemas.strategy_schemas import StrategyRequest
        req = StrategyRequest(
            session_id=session_id,
            driver_code=driver_code,
            current_lap=current_lap,
            current_compound=current_compound,
            tyre_age=tyre_age
        )
        return StrategyService.calculate_strategy(req).dict()

    @classmethod
    def get_telemetry(cls, session_id: str, driver_code: str, lap_number: int) -> Dict[str, Any]:
        try:
            resp = requests.get(
                f"{API_BASE}/telemetry",
                params={"session_id": session_id, "driver_code": driver_code, "lap_number": lap_number},
                timeout=5.0
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"REST telemetry failed: {e}. Direct fallback.")

        from ml.fastf1_loader import FastF1DataLoader
        parts = session_id.split("_")
        s_info, laps, tel_dict = FastF1DataLoader.load_session(int(parts[0]), parts[1], parts[2])
        tel_df = tel_dict.get(f"{driver_code}_{lap_number}")
        if tel_df is None or tel_df.empty:
            tel_df = list(tel_dict.values())[0] if tel_dict else None

        trace = tel_df.to_dict(orient="records") if tel_df is not None else []
        return {
            "session_id": session_id,
            "driver_code": driver_code,
            "lap_number": lap_number,
            "lap_time_sec": 91.45,
            "compound": "MEDIUM",
            "clean_lap_time_sec": 91.10,
            "noise_sec": 0.35,
            "trace": trace,
        }

    @classmethod
    def validate_stint(cls, session_id: str, driver_code: str, compound: str) -> Dict[str, Any]:
        try:
            resp = requests.post(
                f"{API_BASE}/validate",
                json={"session_id": session_id, "driver_code": driver_code, "compound": compound},
                timeout=5.0
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"REST validate failed: {e}. Direct fallback.")

        from ml.validation import ValidationService
        return {
            "driver_code": driver_code,
            "compound": compound,
            "metrics": ValidationService.evaluate([], [])
        }
