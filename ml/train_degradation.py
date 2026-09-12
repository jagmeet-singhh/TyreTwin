import os
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from backend.config import settings

logger = logging.getLogger("TyreTwin.DegradationModel")


class TyreDegradationModel:
    """Model 2: Gaussian Process Regression modeling true degradation curves with uncertainty for car & 4 corners."""

    def __init__(self):
        self.gp_models: Dict[str, GaussianProcessRegressor] = {}
        self.gp_delta_models: Dict[str, GaussianProcessRegressor] = {}
        self.gp_corner_models: Dict[Tuple[str, str], GaussianProcessRegressor] = {}
        self.base_paces: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {
            "trained_seasons": [],
            "sample_counts": {},
            "metrics": {},
        }

    def fit_compound_delta(self, compound: str, tyre_ages: np.ndarray, deg_deltas: np.ndarray):
        """Fits Gaussian Process on normalized degradation delta (seconds degraded from fresh tyre pace)."""
        if len(tyre_ages) < 3:
            return

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=7.0, nu=2.5, length_scale_bounds=(1.0, 40.0))
            + WhiteKernel(noise_level=0.02, noise_level_bounds=(1e-4, 1e0))
        )
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42)
        X = tyre_ages.reshape(-1, 1)
        gp.fit(X, deg_deltas)
        
        comp_key = compound.upper()
        self.gp_delta_models[comp_key] = gp
        self.metadata["sample_counts"][comp_key] = len(tyre_ages)
        logger.info(f"Fitted GP Delta Model for {comp_key} across {len(tyre_ages)} multi-season stint laps.")

    def fit_corner_delta(self, compound: str, corner: str, tyre_ages: np.ndarray, deg_deltas: np.ndarray):
        """Fits Gaussian Process on corner-specific degradation delta for FL, FR, RL, or RR."""
        if len(tyre_ages) < 3:
            return

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=7.0, nu=2.5, length_scale_bounds=(1.0, 40.0))
            + WhiteKernel(noise_level=0.02, noise_level_bounds=(1e-4, 1e0))
        )
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42)
        X = tyre_ages.reshape(-1, 1)
        gp.fit(X, deg_deltas)
        
        comp_key = compound.upper()
        corn_key = corner.upper()
        self.gp_corner_models[(comp_key, corn_key)] = gp
        logger.info(f"Fitted GP Corner Model for ({comp_key}, {corn_key}) across {len(tyre_ages)} stint laps.")

    def fit_compound(self, compound: str, tyre_ages: np.ndarray, clean_lap_times: np.ndarray):
        """Fits Gaussian Process on raw lap times (stint-specific)."""
        if len(tyre_ages) < 3:
            return

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=6.0, nu=2.5, length_scale_bounds=(1.0, 30.0))
            + WhiteKernel(noise_level=0.02, noise_level_bounds=(1e-4, 1e0))
        )
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42)
        X = tyre_ages.reshape(-1, 1)
        gp.fit(X, clean_lap_times)
        
        comp_key = compound.upper()
        self.gp_models[comp_key] = gp
        self.base_paces[comp_key] = float(np.min(clean_lap_times))
        logger.info(f"Fitted Gaussian Process for {comp_key} across {len(tyre_ages)} stint laps.")

    def predict_stint(
        self, compound: str, max_age: int = 35, base_pace: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        compound = compound.upper()
        ages = np.arange(1, max_age + 1).reshape(-1, 1)
        base = base_pace if base_pace is not None else self.base_paces.get(compound, 90.0)

        # 1. Prefer unified multi-season delta GP model
        if compound in self.gp_delta_models:
            gp = self.gp_delta_models[compound]
            delta_mean, y_sigma = gp.predict(ages, return_std=True)
            delta_mean = np.maximum(0.0, delta_mean)
            y_mean = base + delta_mean
        # 2. Stint-specific absolute GP model
        elif compound in self.gp_models:
            gp = self.gp_models[compound]
            y_mean, y_sigma = gp.predict(ages, return_std=True)
            if base_pace is not None and compound in self.base_paces:
                shift = base_pace - self.base_paces[compound]
                y_mean = y_mean + shift
        # 3. Physics-based heuristic prior
        else:
            deg_rates = {"SOFT": 0.088, "MEDIUM": 0.054, "HARD": 0.032, "INTERMEDIATE": 0.07, "WET": 0.05}
            rate = deg_rates.get(compound, 0.055)
            cliff_lap = 18 if compound == "SOFT" else (28 if compound == "MEDIUM" else 38)
            cliff_effect = np.where(ages.flatten() > cliff_lap, (ages.flatten() - cliff_lap) ** 2 * 0.02, 0.0)
            y_mean = base + (ages.flatten() * rate) + cliff_effect
            y_sigma = 0.05 + (ages.flatten() * 0.008)

        lower_ci = y_mean - (2.0 * y_sigma)
        upper_ci = y_mean + (2.0 * y_sigma)
        return ages.flatten(), y_mean, lower_ci, upper_ci, y_sigma

    def predict_corner_stint(
        self, compound: str, corner: str, max_age: int = 35, base_pace: Optional[float] = None, load_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predicts degradation curve for an individual tyre corner (FL, FR, RL, RR)."""
        compound = compound.upper()
        corner = corner.upper()
        ages = np.arange(1, max_age + 1).reshape(-1, 1)
        base = base_pace if base_pace is not None else self.base_paces.get(compound, 90.0)

        key = (compound, corner)
        if key in self.gp_corner_models:
            gp = self.gp_corner_models[key]
            delta_mean, y_sigma = gp.predict(ages, return_std=True)
            delta_mean = np.maximum(0.0, delta_mean)
            y_mean = base + delta_mean
        elif compound in self.gp_delta_models:
            gp = self.gp_delta_models[compound]
            delta_mean, y_sigma = gp.predict(ages, return_std=True)
            delta_mean = np.maximum(0.0, delta_mean) * load_factor
            y_mean = base + delta_mean
            y_sigma = y_sigma * (0.8 + 0.2 * load_factor)
        else:
            deg_rates = {"SOFT": 0.088, "MEDIUM": 0.054, "HARD": 0.032, "INTERMEDIATE": 0.07, "WET": 0.05}
            rate = deg_rates.get(compound, 0.055) * load_factor
            cliff_lap = max(5, int((18 if compound == "SOFT" else (28 if compound == "MEDIUM" else 38)) / max(0.5, load_factor)))
            cliff_effect = np.where(ages.flatten() > cliff_lap, (ages.flatten() - cliff_lap) ** 2 * 0.02, 0.0)
            y_mean = base + (ages.flatten() * rate) + cliff_effect
            y_sigma = (0.05 + (ages.flatten() * 0.008)) * (0.8 + 0.2 * load_factor)

        lower_ci = y_mean - (2.0 * y_sigma)
        upper_ci = y_mean + (2.0 * y_sigma)
        return ages.flatten(), y_mean, lower_ci, upper_ci, y_sigma

    def save(self, filepath: str = settings.DEGRADATION_MODEL_PATH):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Convert tuple keys to string keys for clean serialization
        corner_models_dict = {f"{c[0]}_{c[1]}": m for c, m in self.gp_corner_models.items()}
        joblib.dump({
            "gp_models": self.gp_models,
            "gp_delta_models": self.gp_delta_models,
            "gp_corner_models": corner_models_dict,
            "base_paces": self.base_paces,
            "metadata": self.metadata
        }, filepath)
        logger.info(f"Saved TyreDegradationModel to {filepath}")

    @classmethod
    def load(cls, filepath: str = settings.DEGRADATION_MODEL_PATH) -> "TyreDegradationModel":
        inst = cls()
        if Path(filepath).exists():
            data = joblib.load(filepath)
            inst.gp_models = data.get("gp_models", {})
            inst.gp_delta_models = data.get("gp_delta_models", {})
            inst.base_paces = data.get("base_paces", {})
            inst.metadata = data.get("metadata", {})
            
            raw_corners = data.get("gp_corner_models", {})
            for k, m in raw_corners.items():
                if "_" in k:
                    parts = k.split("_", 1)
                    inst.gp_corner_models[(parts[0], parts[1])] = m
                elif isinstance(k, tuple):
                    inst.gp_corner_models[k] = m
                    
            logger.info(f"Loaded TyreDegradationModel from {filepath}")
        return inst
