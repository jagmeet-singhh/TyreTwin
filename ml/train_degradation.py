import os
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from backend.config import settings

logger = logging.getLogger("TyreIQ.DegradationModel")


class TyreDegradationModel:
    """Model 2: Gaussian Process Regression modeling true degradation curves with uncertainty."""

    def __init__(self):
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=6.0, nu=2.5, length_scale_bounds=(1.0, 30.0))
            + WhiteKernel(noise_level=0.03, noise_level_bounds=(1e-4, 1e0))
        )
        self.gp_models: Dict[str, GaussianProcessRegressor] = {}
        self.base_paces: Dict[str, float] = {}

    def fit_compound(self, compound: str, tyre_ages: np.ndarray, clean_lap_times: np.ndarray):
        if len(tyre_ages) < 3:
            return

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=6.0, nu=2.5, length_scale_bounds=(1.0, 30.0))
            + WhiteKernel(noise_level=0.02, noise_level_bounds=(1e-4, 1e0))
        )
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)
        X = tyre_ages.reshape(-1, 1)
        gp.fit(X, clean_lap_times)
        
        self.gp_models[compound.upper()] = gp
        self.base_paces[compound.upper()] = float(np.min(clean_lap_times))
        logger.info(f"Fitted Gaussian Process for {compound.upper()} across {len(tyre_ages)} stint laps.")

    def predict_stint(
        self, compound: str, max_age: int = 35, base_pace: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        compound = compound.upper()
        ages = np.arange(1, max_age + 1).reshape(-1, 1)

        if compound in self.gp_models:
            gp = self.gp_models[compound]
            y_mean, y_sigma = gp.predict(ages, return_std=True)
        else:
            # Physics-based heuristic prior if compound not yet fitted
            deg_rates = {"SOFT": 0.088, "MEDIUM": 0.054, "HARD": 0.032, "INTERMEDIATE": 0.07, "WET": 0.05}
            rate = deg_rates.get(compound, 0.055)
            base = base_pace or 90.0
            
            # Linear + exponential cliff onset
            cliff_lap = 18 if compound == "SOFT" else (28 if compound == "MEDIUM" else 38)
            cliff_effect = np.where(ages.flatten() > cliff_lap, (ages.flatten() - cliff_lap) ** 2 * 0.02, 0.0)
            
            y_mean = base + (ages.flatten() * rate) + cliff_effect
            y_sigma = 0.05 + (ages.flatten() * 0.008)  # expanding uncertainty

        lower_ci = y_mean - (2.0 * y_sigma)
        upper_ci = y_mean + (2.0 * y_sigma)
        return ages.flatten(), y_mean, lower_ci, upper_ci, y_sigma

    def save(self, filepath: str = settings.DEGRADATION_MODEL_PATH):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"gp_models": self.gp_models, "base_paces": self.base_paces}, filepath)
        logger.info(f"Saved TyreDegradationModel to {filepath}")

    @classmethod
    def load(cls, filepath: str = settings.DEGRADATION_MODEL_PATH) -> "TyreDegradationModel":
        inst = cls()
        if Path(filepath).exists():
            data = joblib.load(filepath)
            inst.gp_models = data["gp_models"]
            inst.base_paces = data["base_paces"]
            logger.info(f"Loaded TyreDegradationModel from {filepath}")
        return inst
