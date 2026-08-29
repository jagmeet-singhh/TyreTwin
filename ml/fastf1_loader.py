import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from backend.config import settings

logger = logging.getLogger("TyreIQ.FastF1Loader")

try:
    import fastf1
    # Enable cache if configured
    if settings.ENABLE_FASTF1_CACHE:
        fastf1.Cache.enable_cache(settings.FASTF1_CACHE_DIR)
except Exception as e:
    logger.warning(f"FastF1 cache initialization warning: {e}")

BENCHMARK_CONFIGS = {
    "2024_Bahrain_FP2": {
        "season": 2024,
        "grand_prix": "Bahrain",
        "session_name": "FP2",
        "track_name": "Bahrain International Circuit",
        "track_length_km": 5.412,
        "air_temp_c": 24.5,
        "track_temp_c": 33.8,
        "total_laps": 57,
        "base_lap_sec": 91.5,  # 1:31.500
    },
    "2024_Silverstone_FP2": {
        "season": 2024,
        "grand_prix": "Silverstone",
        "session_name": "FP2",
        "track_name": "Silverstone Circuit",
        "track_length_km": 5.891,
        "air_temp_c": 19.2,
        "track_temp_c": 28.4,
        "total_laps": 52,
        "base_lap_sec": 86.8,  # 1:26.800
    },
    "2024_Monza_FP2": {
        "season": 2024,
        "grand_prix": "Monza",
        "session_name": "FP2",
        "track_name": "Autodromo Nazionale Monza",
        "track_length_km": 5.793,
        "air_temp_c": 28.0,
        "track_temp_c": 41.5,
        "total_laps": 53,
        "base_lap_sec": 81.2,  # 1:21.200
    },
    "2024_Suzuka_FP2": {
        "season": 2024,
        "grand_prix": "Suzuka",
        "session_name": "FP2",
        "track_name": "Suzuka International Racing Course",
        "track_length_km": 5.807,
        "air_temp_c": 17.5,
        "track_temp_c": 22.0,
        "total_laps": 53,
        "base_lap_sec": 90.0,  # 1:30.000
    },
}

DRIVERS_METADATA = [
    {"code": "VER", "number": 1, "full_name": "Max Verstappen", "team_name": "Red Bull Racing", "team_color": "#3671C6"},
    {"code": "HAM", "number": 44, "full_name": "Lewis Hamilton", "team_name": "Mercedes-AMG Petronas", "team_color": "#6CD3BF"},
    {"code": "LEC", "number": 16, "full_name": "Charles Leclerc", "team_name": "Scuderia Ferrari", "team_color": "#E80020"},
    {"code": "NOR", "number": 4, "full_name": "Lando Norris", "team_name": "McLaren F1 Team", "team_color": "#FF8000"},
    {"code": "SAI", "number": 55, "full_name": "Carlos Sainz", "team_name": "Scuderia Ferrari", "team_color": "#E80020"},
    {"code": "RUS", "number": 63, "full_name": "George Russell", "team_name": "Mercedes-AMG Petronas", "team_color": "#6CD3BF"},
    {"code": "PIA", "number": 81, "full_name": "Oscar Piastri", "team_name": "McLaren F1 Team", "team_color": "#FF8000"},
    {"code": "PER", "number": 11, "full_name": "Sergio Perez", "team_name": "Red Bull Racing", "team_color": "#3671C6"},
    {"code": "ALO", "number": 14, "full_name": "Fernando Alonso", "team_name": "Aston Martin", "team_color": "#229971"},
    {"code": "ALB", "number": 23, "full_name": "Alexander Albon", "team_name": "Williams Racing", "team_color": "#64C4FF"},
]


class FastF1DataLoader:
    """Loads F1 data from FastF1 with caching and physics-accurate synthetic benchmark fallback."""

    @classmethod
    def load_session(
        cls, season: int = 2024, grand_prix: str = "Bahrain", session_name: str = "FP2", force_refresh: bool = False
    ) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, pd.DataFrame]]:
        session_id = f"{season}_{grand_prix}_{session_name}"
        benchmark_file = Path(settings.BENCHMARK_DIR) / f"{session_id}.json"

        # Check local pre-bundled benchmark cache first
        if not force_refresh and benchmark_file.exists():
            try:
                with open(benchmark_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                session_info = cached_data["session_info"]
                laps_df = pd.DataFrame(cached_data["laps"])
                telemetry_dict = {
                    k: pd.DataFrame(v) for k, v in cached_data["telemetry"].items()
                }
                logger.info(f"Loaded {session_id} from benchmark cache.")
                return session_info, laps_df, telemetry_dict
            except Exception as e:
                logger.warning(f"Failed to read local cache: {e}. Re-generating.")

        # Attempt to load via FastF1 API
        try:
            logger.info(f"Attempting FastF1 API load for {season} {grand_prix} {session_name}...")
            f1_session = fastf1.get_session(season, grand_prix, session_name)
            f1_session.load(laps=True, telemetry=True, weather=True)
            
            session_info = {
                "id": session_id,
                "season": season,
                "grand_prix": grand_prix,
                "session_name": session_name,
                "track_name": f1_session.event["Location"],
                "track_length_km": 5.412,
                "air_temp_c": float(f1_session.weather_data["AirTemp"].mean() if not f1_session.weather_data.empty else 25.0),
                "track_temp_c": float(f1_session.weather_data["TrackTemp"].mean() if not f1_session.weather_data.empty else 35.0),
                "humidity_pct": float(f1_session.weather_data["Humidity"].mean() if not f1_session.weather_data.empty else 40.0),
                "rainfall": bool(f1_session.weather_data["Rainfall"].any() if not f1_session.weather_data.empty else False),
                "total_laps": 57,
            }

            raw_laps = f1_session.laps
            laps_records = []
            telemetry_dict = {}

            for idx, lap in raw_laps.iterlaps():
                if pd.isna(lap["LapTime"]):
                    continue
                lap_sec = lap["LapTime"].total_seconds()
                driver = lap["Driver"]
                lap_num = int(lap["LapNumber"])
                compound = str(lap["Compound"]).upper() if pd.notna(lap["Compound"]) else "MEDIUM"
                tyre_age = int(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else lap_num
                
                s1 = lap["Sector1Time"].total_seconds() if pd.notna(lap["Sector1Time"]) else lap_sec * 0.32
                s2 = lap["Sector2Time"].total_seconds() if pd.notna(lap["Sector2Time"]) else lap_sec * 0.40
                s3 = lap["Sector3Time"].total_seconds() if pd.notna(lap["Sector3Time"]) else lap_sec * 0.28

                is_pit_in = pd.notna(lap["PitInTime"])
                is_pit_out = pd.notna(lap["PitOutTime"])
                track_status = str(lap["TrackStatus"]) if pd.notna(lap["TrackStatus"]) else "1"

                laps_records.append({
                    "session_id": session_id,
                    "driver_code": driver,
                    "lap_number": lap_num,
                    "lap_time_sec": lap_sec,
                    "sector1_sec": s1,
                    "sector2_sec": s2,
                    "sector3_sec": s3,
                    "compound": compound,
                    "tyre_age_lap": tyre_age,
                    "stint_number": int(lap["Stint"]) if pd.notna(lap["Stint"]) else 1,
                    "is_valid": bool(lap["IsAccurate"]) if pd.notna(lap["IsAccurate"]) else (not is_pit_in and not is_pit_out),
                    "is_pit_in": is_pit_in,
                    "is_pit_out": is_pit_out,
                    "track_status": track_status,
                })

                # Sample telemetry
                try:
                    tel = lap.get_telemetry()
                    if not tel.empty:
                        sub_tel = tel.iloc[::max(1, len(tel) // 100)]  # 100 points per lap
                        tel_records = []
                        for s_idx, (_, row) in enumerate(sub_tel.iterrows()):
                            tel_records.append({
                                "sample_index": s_idx,
                                "distance_m": float(row.get("Distance", s_idx * 54.0)),
                                "speed_kmh": float(row.get("Speed", 220.0)),
                                "throttle_pct": float(row.get("Throttle", 85.0)),
                                "brake_pct": float(row.get("Brake", 0.0) * 100.0 if row.get("Brake", 0) <= 1 else row.get("Brake", 0)),
                                "rpm": int(row.get("RPM", 11500)),
                                "gear": int(row.get("nGear", 6)),
                                "drs": int(row.get("DRS", 0)),
                                "time_sec": float(row.get("Time").total_seconds() if pd.notna(row.get("Time")) else s_idx * 0.9),
                            })
                        telemetry_dict[f"{driver}_{lap_num}"] = pd.DataFrame(tel_records)
                except Exception:
                    pass

            laps_df = pd.DataFrame(laps_records)
            cls._save_benchmark_cache(session_id, session_info, laps_df, telemetry_dict)
            return session_info, laps_df, telemetry_dict

        except Exception as e:
            logger.warning(f"FastF1 API unavailable or offline ({e}). Generating high-fidelity benchmark telemetry.")
            return cls.generate_benchmark_session(session_id)

    @classmethod
    def generate_benchmark_session(
        cls, session_id: str = "2024_Bahrain_FP2"
    ) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Generates physics-based motorsport telemetry with real F1 tire compounds and noise."""
        cfg = BENCHMARK_CONFIGS.get(session_id, BENCHMARK_CONFIGS["2024_Bahrain_FP2"])
        base_lap = cfg["base_lap_sec"]
        
        session_info = {
            "id": session_id,
            "season": cfg["season"],
            "grand_prix": cfg["grand_prix"],
            "session_name": cfg["session_name"],
            "track_name": cfg["track_name"],
            "track_length_km": cfg["track_length_km"],
            "air_temp_c": cfg["air_temp_c"],
            "track_temp_c": cfg["track_temp_c"],
            "humidity_pct": 42.0,
            "rainfall": False,
            "total_laps": cfg["total_laps"],
        }

        # Compound characteristics (base delta, degradation rate sec/lap, cliff lap)
        compound_specs = {
            "SOFT": {"delta": -0.85, "deg_rate": 0.088, "cliff_lap": 18, "wear_factor": 1.4},
            "MEDIUM": {"delta": 0.00, "deg_rate": 0.054, "cliff_lap": 28, "wear_factor": 1.0},
            "HARD": {"delta": 0.75, "deg_rate": 0.032, "cliff_lap": 38, "wear_factor": 0.7},
        }

        driver_deltas = {
            "VER": -0.35, "LEC": -0.28, "NOR": -0.22, "HAM": -0.15,
            "SAI": -0.10, "RUS": -0.08, "PIA": 0.05, "PER": 0.12,
            "ALO": 0.15, "ALB": 0.45
        }

        np.random.seed(42)
        laps_records = []
        telemetry_dict = {}

        for driver_meta in DRIVERS_METADATA:
            driver = driver_meta["code"]
            drv_delta = driver_deltas.get(driver, 0.0)

            # Generate 2 realistic practice stints
            # Stint 1: SOFT (12 laps)
            # Stint 2: MEDIUM (16 laps)
            stints_plan = [
                ("SOFT", 1, 14),
                ("MEDIUM", 15, 32),
            ]

            for compound, start_lap, end_lap in stints_plan:
                spec = compound_specs[compound]
                stint_num = 1 if start_lap == 1 else 2

                for lap_num in range(start_lap, end_lap + 1):
                    tyre_age = lap_num - start_lap + 1
                    is_in_lap = (lap_num == end_lap)
                    is_out_lap = (lap_num == start_lap)

                    # Fuel mass burn: 105kg at race start, in FP ~ 45kg decreasing
                    fuel_kg = max(5.0, 45.0 - (lap_num * 1.7))
                    fuel_noise = (fuel_kg / 10.0) * settings.FUEL_EFFECT_SEC_PER_10KG

                    # Track evolution: track gets 0.4s faster over session
                    track_evo_idx = lap_num / 40.0
                    track_evo_gain = -0.45 * np.log1p(track_evo_idx * 1.7)

                    # True tyre degradation: exponential onset near cliff
                    cliff = spec["cliff_lap"]
                    cliff_effect = max(0.0, (tyre_age - cliff) ** 2 * 0.025) if tyre_age > cliff else 0.0
                    true_deg = (tyre_age * spec["deg_rate"]) + cliff_effect

                    # Driver push aggression & random variance
                    push_idx = np.random.uniform(0.78, 0.96)
                    push_noise = (1.0 - push_idx) * 0.40

                    # Traffic injection (15% chance of traffic lap)
                    has_traffic = (np.random.rand() < 0.14) and not is_out_lap and not is_in_lap
                    traffic_noise = np.random.uniform(0.35, 1.40) if has_traffic else 0.0

                    # Base clean lap time
                    clean_time = base_lap + drv_delta + spec["delta"] + true_deg
                    
                    # Total observed lap time = Clean + External Noise
                    total_noise = fuel_noise + track_evo_gain + push_noise + traffic_noise + np.random.normal(0, 0.04)
                    
                    if is_out_lap:
                        lap_time = clean_time + 15.0  # Out lap delta
                        is_valid = False
                    elif is_in_lap:
                        lap_time = clean_time + 8.0   # In lap delta
                        is_valid = False
                    else:
                        lap_time = clean_time + total_noise
                        is_valid = True

                    s1 = lap_time * 0.315 + np.random.normal(0, 0.03)
                    s2 = lap_time * 0.410 + np.random.normal(0, 0.03)
                    s3 = lap_time - s1 - s2

                    laps_records.append({
                        "session_id": session_id,
                        "driver_code": driver,
                        "lap_number": lap_num,
                        "lap_time_sec": round(lap_time, 3),
                        "sector1_sec": round(s1, 3),
                        "sector2_sec": round(s2, 3),
                        "sector3_sec": round(s3, 3),
                        "compound": compound,
                        "tyre_age_lap": tyre_age,
                        "stint_number": stint_num,
                        "is_valid": is_valid,
                        "is_pit_in": is_in_lap,
                        "is_pit_out": is_out_lap,
                        "track_status": "1",
                        "fuel_kg": round(fuel_kg, 1),
                        "driver_push_index": round(push_idx, 3),
                        "traffic_flag": has_traffic,
                        "track_evolution_index": round(track_evo_idx, 3),
                        "clean_lap_time_sec": round(clean_time, 3),
                        "predicted_noise_sec": round(total_noise, 3),
                    })

                    # Generate rich telemetry profile for valid laps
                    if is_valid and (lap_num in [start_lap + 2, start_lap + 5, end_lap - 2]):
                        tel_key = f"{driver}_{lap_num}"
                        tel_records = cls._generate_lap_telemetry(cfg["track_length_km"] * 1000, lap_time, tyre_age)
                        telemetry_dict[tel_key] = pd.DataFrame(tel_records)

        laps_df = pd.DataFrame(laps_records)
        cls._save_benchmark_cache(session_id, session_info, laps_df, telemetry_dict)
        return session_info, laps_df, telemetry_dict

    @classmethod
    def _generate_lap_telemetry(cls, track_length_m: float, lap_time_sec: float, tyre_age: int) -> List[Dict[str, Any]]:
        """Generates synchronized 100-point telemetry trace for F1 telemetry explorer."""
        n_points = 100
        distances = np.linspace(0, track_length_m, n_points)
        time_points = np.linspace(0, lap_time_sec, n_points)
        
        # Base speed profile with 3 main straights and 15 corners
        base_speed = 220 + 80 * np.sin(distances / 350.0) + 40 * np.cos(distances / 700.0)
        # Tyre wear degrades cornering apex minimum speed
        wear_drop = min(12.0, tyre_age * 0.45)
        speeds = np.clip(base_speed - wear_drop, 65.0, 340.0)

        # Throttle & Brake
        throttles = np.where(speeds > 200, 100.0, np.clip((speeds - 65) / 135.0 * 100, 0, 100))
        brakes = np.where(throttles < 30, np.clip((100 - throttles) * 1.1, 0, 100), 0.0)
        
        # RPM & Gears
        gears = np.clip(np.floor(speeds / 45) + 1, 2, 8).astype(int)
        rpms = np.clip(10000 + (speeds % 45) * 110, 9500, 13200).astype(int)
        
        # DRS zones (typically main straight: first 600m and 3200-3800m)
        drs = np.where((distances < 600) | ((distances > 3200) & (distances < 3800)), 1, 0)

        records = []
        for i in range(n_points):
            records.append({
                "sample_index": i,
                "distance_m": round(float(distances[i]), 1),
                "speed_kmh": round(float(speeds[i]), 1),
                "throttle_pct": round(float(throttles[i]), 1),
                "brake_pct": round(float(brakes[i]), 1),
                "rpm": int(rpms[i]),
                "gear": int(gears[i]),
                "drs": int(drs[i]),
                "time_sec": round(float(time_points[i]), 3),
            })
        return records

    @classmethod
    def _save_benchmark_cache(
        cls, session_id: str, session_info: Dict[str, Any], laps_df: pd.DataFrame, telemetry_dict: Dict[str, pd.DataFrame]
    ):
        try:
            cache_file = Path(settings.BENCHMARK_DIR) / f"{session_id}.json"
            data = {
                "session_info": session_info,
                "laps": laps_df.to_dict(orient="records"),
                "telemetry": {k: v.to_dict(orient="records") for k, v in telemetry_dict.items()},
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved benchmark cache for {session_id}")
        except Exception as e:
            logger.error(f"Failed to cache benchmark: {e}")
