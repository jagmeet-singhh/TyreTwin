import os
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from backend.config import settings
from ml.feature_engineering import FeatureEngineer

logger = logging.getLogger("TyreIQ.NoiseModel")


class NoiseIsolationModel:
    """Model 1: Predicts external non-tyre noise from lap telemetry features."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.85,
            random_state=42
        )
        self.feature_cols = FeatureEngineer.get_feature_columns()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y_noise: pd.Series):
        X_feats = X[self.feature_cols].copy()
        self.model.fit(X_feats, y_noise)
        self.is_fitted = True
        logger.info("NoiseIsolationModel fitted successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            # Fallback heuristic calculation if model not yet saved to disk
            return X["engineered_noise_sec"].values if "engineered_noise_sec" in X.columns else np.zeros(len(X))
        X_feats = X[self.feature_cols].copy()
        return self.model.predict(X_feats)

    def save(self, filepath: str = settings.NOISE_MODEL_PATH):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols, "is_fitted": self.is_fitted}, filepath)
        logger.info(f"Saved NoiseIsolationModel to {filepath}")

    @classmethod
    def load(cls, filepath: str = settings.NOISE_MODEL_PATH) -> "NoiseIsolationModel":
        inst = cls()
        if Path(filepath).exists():
            data = joblib.load(filepath)
            inst.model = data["model"]
            inst.feature_cols = data["feature_cols"]
            inst.is_fitted = data["is_fitted"]
            logger.info(f"Loaded NoiseIsolationModel from {filepath}")
        return inst
