import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from backend.database import crud
from ml.fastf1_loader import FastF1DataLoader, DRIVERS_METADATA
from ml.preprocessing import DataPreprocessor

logger = logging.getLogger("TyreTwin.DataService")


class DataService:
    """Service managing session ingestion, caching, and database synchronisation."""

    @classmethod
    def load_and_sync_session(
        cls, db: Session, season: int, grand_prix: str, session_name: str, force_refresh: bool = False
    ) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, pd.DataFrame]]:
        session_info, laps_df, telemetry_dict = FastF1DataLoader.load_session(
            season, grand_prix, session_name, force_refresh=force_refresh
        )

        # 1. Sync Session Table
        crud.create_or_update_session(db, session_info)

        # 2. Sync Drivers Table
        for d in DRIVERS_METADATA:
            crud.create_or_update_driver(db, d)

        # 3. Clean Laps & Save if not present
        existing_laps = crud.get_driver_laps(db, session_info["id"], "VER")
        if not existing_laps:
            cleaned_df = DataPreprocessor.clean_laps_data(laps_df)
            laps_records = cleaned_df.to_dict(orient="records")
            try:
                crud.save_laps_batch(db, laps_records)
                logger.info(f"Persisted {len(laps_records)} laps to DB.")
            except Exception as e:
                logger.warning(f"Error persisting laps: {e}")

        return session_info, laps_df, telemetry_dict
