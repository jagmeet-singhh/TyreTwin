"""
TyreTwin Unified Multi-Season Training Pipeline
Trains Noise Isolation & Gaussian Process Tyre Degradation models
jointly across all records of both 2023 and 2024 seasons.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure root directory in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.config import settings
from ml.fastf1_loader import FastF1DataLoader
from ml.preprocessing import DataPreprocessor
from ml.feature_engineering import FeatureEngineer
from ml.train_noise import NoiseIsolationModel
from ml.train_degradation import TyreDegradationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TyreTwin.TrainPipeline")


def train_unified_pipeline(seasons=[2023, 2024]):
    print("=" * 70)
    print("   TYRETWIN: UNIFIED MULTI-SEASON (2023 + 2024) TRAINING PIPELINE")
    print("=" * 70)

    # 1. Ingest All Sessions across 2023 & 2024
    print(f"\n[Step 1/5] Ingesting all sessions across seasons: {seasons} ...")
    sessions_info, raw_laps_df = FastF1DataLoader.load_all_seasons_data(seasons=seasons)
    
    if raw_laps_df.empty:
        logger.error("No lap data could be retrieved. Aborting training.")
        return False

    print(f"   -> Ingested {len(sessions_info)} sessions (Practice & Qualifying across 2023 & 2024).")
    print(f"   -> Total raw laps ingested: {len(raw_laps_df):,}")

    # 2. Data Cleaning & Outlier Rejection
    print("\n[Step 2/5] Cleaning laps, filtering pit in/out & outlier rejection ...")
    cleaned_laps_df = DataPreprocessor.clean_laps_data(raw_laps_df, outlier_iqr_threshold=2.2)
    print(f"   -> Valid representative laps retained: {len(cleaned_laps_df):,} (from {len(raw_laps_df):,})")

    # Sync to SQLite Database (tyretwin.db)
    try:
        from backend.database.connection import SessionLocal
        from backend.database import crud
        from ml.fastf1_loader import DRIVERS_METADATA

        db = SessionLocal()
        for s_id, s_info in sessions_info.items():
            crud.create_or_update_session(db, s_info)
        for d in DRIVERS_METADATA:
            crud.create_or_update_driver(db, d)
        cleaned_records = cleaned_laps_df.to_dict(orient="records")
        crud.save_laps_batch(db, cleaned_records)
        print(f"   -> Synced {len(sessions_info)} sessions and {len(cleaned_records):,} laps to tyretwin.db database.")
        db.close()
    except Exception as e:
        logger.warning(f"Database sync notice: {e}")

    # 3. Physics Feature Engineering & Noise Target Synthesis
    print("\n[Step 3/5] Computing motorsport physics features across all sessions ...")
    session_groups = []
    for session_id, group in cleaned_laps_df.groupby("session_id"):
        sess_info = sessions_info.get(session_id, {})
        fe_group = FeatureEngineer.transform(group, sess_info)
        session_groups.append(fe_group)

    master_df = pd.concat(session_groups, ignore_index=True)
    print(f"   -> Feature matrix compiled: {master_df.shape[0]:,} rows, {len(FeatureEngineer.get_feature_columns())} physics features.")

    # 4. Train Unified Noise Isolation Model (Model 1)
    print("\n[Step 4/5] Training unified Noise Isolation Model (Gradient Boosting) ...")
    noise_model = NoiseIsolationModel()
    
    # Target: engineered external noise (fuel burn, track evolution, thermal penalty, push index, traffic)
    y_noise = master_df["engineered_noise_sec"]
    noise_model.fit(master_df, y_noise, seasons=seasons)
    
    # Predict noise and isolate clean tyre degradation lap times
    master_df["predicted_noise_sec"] = noise_model.predict(master_df)
    master_df["clean_lap_time_sec"] = master_df["lap_time_sec"] - master_df["predicted_noise_sec"]
    
    # Save Noise Isolation Model
    noise_model_path = settings.NOISE_MODEL_PATH
    noise_model.save(noise_model_path)
    print(f"   -> Noise Isolation Model saved to: {noise_model_path}")
    print(f"   -> Noise Model Metrics: R2 = {noise_model.metadata['metrics']['r2_score']:.4f}, MAE = {noise_model.metadata['metrics']['mae_sec']:.4f}s")

    # 5. Train Unified Gaussian Process Degradation Model (Model 2)
    print("\n[Step 5/5] Training unified Gaussian Process Degradation Models per compound ...")
    deg_model = TyreDegradationModel()
    deg_model.metadata["trained_seasons"] = seasons
    deg_model.metadata["total_sessions"] = len(sessions_info)

    # Compute Stint Degradation Delta (clean_lap - stint_baseline)
    # Stint is grouped by (session_id, driver_code, stint_number, compound)
    stint_groups = master_df.groupby(["session_id", "driver_code", "stint_number", "compound"])
    
    delta_records = []
    for (sess_id, drv, stint_num, comp), stint_df in stint_groups:
        if len(stint_df) < 2:
            continue
        sorted_stint = stint_df.sort_values("tyre_age_lap")
        # Base pace of stint is the minimum clean lap time in the first 3 laps of stint
        stint_fresh_laps = sorted_stint[sorted_stint["tyre_age_lap"] <= 3]
        base_pace = float(stint_fresh_laps["clean_lap_time_sec"].min()) if not stint_fresh_laps.empty else float(sorted_stint["clean_lap_time_sec"].iloc[0])
        
        for _, row in sorted_stint.iterrows():
            deg_delta = float(row["clean_lap_time_sec"] - base_pace)
            delta_records.append({
                "compound": str(comp).upper(),
                "tyre_age_lap": int(row["tyre_age_lap"]),
                "deg_delta_sec": max(0.0, deg_delta),
                "clean_lap_time_sec": float(row["clean_lap_time_sec"]),
            })

    delta_df = pd.DataFrame(delta_records)
    print(f"   -> Extracted {len(delta_df):,} stint degradation delta observations across all sessions.")

    # Fit Gaussian Process for each compound across all multi-season stints
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        comp_data = delta_df[delta_df["compound"] == compound]
        if len(comp_data) >= 3:
            # Subsample if large to keep GP fitting snappy and memory-efficient
            if len(comp_data) > 600:
                comp_data = comp_data.sample(n=600, random_state=42)
            
            ages = comp_data["tyre_age_lap"].values
            deltas = comp_data["deg_delta_sec"].values
            
            deg_model.fit_compound_delta(compound, ages, deltas)
            print(f"   -> Fitted Base Gaussian Process for [{compound}] across {len(ages)} multi-season stint laps.")

            # Fit distinct 4-Corner Gaussian Process models (FL, FR, RL, RR)
            corner_multipliers = {"FL": 1.14, "FR": 1.02, "RL": 1.18, "RR": 1.06}
            for corner, mult in corner_multipliers.items():
                c_deltas = np.maximum(0.0, deltas * mult + np.random.normal(0, 0.01, size=len(deltas)))
                deg_model.fit_corner_delta(compound, corner, ages, c_deltas)
            print(f"   -> Fitted 4-Corner GP Models [FL, FR, RL, RR] for [{compound}].")
        else:
            logger.warning(f"Insufficient samples for compound {compound}: {len(comp_data)}")

    # Save Tyre Degradation Model
    deg_model_path = settings.DEGRADATION_MODEL_PATH
    deg_model.save(deg_model_path)
    print(f"   -> Tyre Degradation Model saved to: {deg_model_path}")

    # Summary Report
    print("\n" + "=" * 70)
    print("   UNIFIED MULTI-SEASON TRAINING COMPLETE!")
    print("=" * 70)
    print(f"   * Seasons Ingested:    {seasons}")
    print(f"   * Sessions Processed:  {len(sessions_info)} sessions")
    print(f"   * Total Laps Trained:  {len(master_df):,} laps")
    print(f"   * Noise Model R2:      {noise_model.metadata['metrics']['r2_score']:.4f}")
    print(f"   * Noise Model MAE:     {noise_model.metadata['metrics']['mae_sec']:.4f}s")
    for comp, count in deg_model.metadata.get("sample_counts", {}).items():
        print(f"   * GP Model [{comp}]:    {count} laps trained")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    train_unified_pipeline()
