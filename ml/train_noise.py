import os
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from sklearn.ensemble import GradientBoostingRegressor
from backend.config import settings
from ml.feature_engineering import FeatureEngineer

logger = logging.getLogger("TyreTwin.NoiseModel")


class NoiseIsolationModel:
    """Model 1: Predicts external non-tyre noise from lap telemetry features."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            random_state=42
        )
        self.feature_cols = FeatureEngineer.get_feature_columns()
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {
            "trained_seasons": [],
            "total_samples": 0,
            "metrics": {},
            "feature_importances": {},
        }

    def fit(self, X: pd.DataFrame, y_noise: pd.Series, seasons: Optional[List[int]] = None):
        X_feats = X[self.feature_cols].copy()
        self.model.fit(X_feats, y_noise)
        self.is_fitted = True

        # Compute training diagnostics
        preds = self.model.predict(X_feats)
        mae = float(np.mean(np.abs(preds - y_noise.values)))
        rmse = float(np.sqrt(np.mean((preds - y_noise.values) ** 2)))
        ss_tot = float(np.sum((y_noise.values - np.mean(y_noise.values)) ** 2))
        ss_res = float(np.sum((y_noise.values - preds) ** 2))
        r2 = float(1.0 - (ss_res / max(1e-6, ss_tot)))

        importances = {col: round(float(imp), 4) for col, imp in zip(self.feature_cols, self.model.feature_importances_)}

        self.metadata = {
            "trained_seasons": seasons or [2023, 2024],
            "total_samples": len(X),
            "metrics": {
                "mae_sec": round(mae, 4),
                "rmse_sec": round(rmse, 4),
                "r2_score": round(max(0.0, r2), 4),
            },
            "feature_importances": importances,
        }
        logger.info(f"NoiseIsolationModel fitted on {len(X)} samples. R2={r2:.3f}, MAE={mae:.4f}s.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            # Fallback heuristic calculation if model not yet saved to disk
            return X["engineered_noise_sec"].values if "engineered_noise_sec" in X.columns else np.zeros(len(X))
        X_feats = X[self.feature_cols].copy()
        return self.model.predict(X_feats)

    def save(self, filepath: str = settings.NOISE_MODEL_PATH):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_cols": self.feature_cols,
            "is_fitted": self.is_fitted,
            "metadata": self.metadata
        }, filepath)
        logger.info(f"Saved NoiseIsolationModel to {filepath}")

    @classmethod
    def load(cls, filepath: str = settings.NOISE_MODEL_PATH) -> "NoiseIsolationModel":
        inst = cls()
        if Path(filepath).exists():
            data = joblib.load(filepath)
            inst.model = data["model"]
            inst.feature_cols = data["feature_cols"]
            inst.is_fitted = data["is_fitted"]
            inst.metadata = data.get("metadata", {})
            logger.info(f"Loaded NoiseIsolationModel from {filepath}")
        return inst
