import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional

logger = logging.getLogger("TyreTwin.Preprocessing")


class DataPreprocessor:
    """Cleans raw F1 lap telemetry, removes outliers, and filters non-representative laps."""

    @classmethod
    def clean_laps_data(
        cls, laps_df: pd.DataFrame, outlier_iqr_threshold: float = 2.0
    ) -> pd.DataFrame:
        if laps_df.empty:
            return laps_df

        df = laps_df.copy()

        # 1. Filter Pit In / Out Laps
        if "is_pit_in" in df.columns:
            df = df[~df["is_pit_in"]]
        if "is_pit_out" in df.columns:
            df = df[~df["is_pit_out"]]

        # 2. Filter Track Status (Only Green Flag track status '1')
        if "track_status" in df.columns:
            df = df[df["track_status"].astype(str).str.contains("1")]

        # 3. Filter explicit is_valid flag
        if "is_valid" in df.columns:
            df = df[df["is_valid"] == True]

        # 4. Outlier Rejection per Driver and Compound using IQR
        cleaned_groups = []
        for (driver, compound), group in df.groupby(["driver_code", "compound"]):
            if len(group) < 4:
                cleaned_groups.append(group)
                continue
            
            q1 = group["lap_time_sec"].quantile(0.25)
            q3 = group["lap_time_sec"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (outlier_iqr_threshold * iqr)
            upper_bound = q3 + (outlier_iqr_threshold * iqr)
            
            valid_mask = (group["lap_time_sec"] >= lower_bound) & (group["lap_time_sec"] <= upper_bound)
            cleaned_groups.append(group[valid_mask])

        if cleaned_groups:
            result_df = pd.concat(cleaned_groups, ignore_index=True)
            logger.info(f"Cleaned laps: {len(laps_df)} -> {len(result_df)} representative laps.")
            return result_df.sort_values(by=["driver_code", "lap_number"]).reset_index(drop=True)

        return df
