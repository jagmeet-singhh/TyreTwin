import pytest
import numpy as np
import pandas as pd
from ml.train_noise import NoiseIsolationModel
from ml.train_degradation import TyreDegradationModel

def test_noise_isolation_model():
    model = NoiseIsolationModel()
    df = pd.DataFrame({
        "fuel_kg": [45.0, 40.0, 35.0, 30.0, 25.0],
        "track_evolution_index": [0.1, 0.2, 0.3, 0.4, 0.5],
        "driver_push_index": [0.9, 0.88, 0.91, 0.89, 0.92],
        "traffic_penalty_sec": [0.0, 0.65, 0.0, 0.0, 0.0],
        "thermal_delta_c": [2.0, 2.0, 2.5, 2.5, 3.0],
        "engineered_noise_sec": [0.4, 1.0, 0.35, 0.3, 0.25]
    })
    y = df["engineered_noise_sec"]
    model.fit(df, y)
    preds = model.predict(df)
    assert len(preds) == 5
    assert preds[1] > preds[0]  # Traffic lap has higher noise

def test_gp_degradation_model():
    deg_model = TyreDegradationModel()
    ages = np.array([1, 3, 5, 7, 9, 11, 13])
    paces = np.array([91.2, 91.3, 91.45, 91.6, 91.8, 92.1, 92.5])
    deg_model.fit_compound("SOFT", ages, paces)
    
    stint_ages, y_mean, lower, upper, sigmas = deg_model.predict_stint("SOFT", max_age=15)
    assert len(y_mean) == 15
    assert np.all(upper >= y_mean)
    assert np.all(lower <= y_mean)
    assert np.all(sigmas > 0)


def test_gp_degradation_delta_model():
    deg_model = TyreDegradationModel()
    ages = np.array([1, 2, 4, 6, 8, 10, 12, 14])
    deltas = np.array([0.0, 0.05, 0.18, 0.35, 0.58, 0.88, 1.25, 1.70])
    deg_model.fit_compound_delta("MEDIUM", ages, deltas)
    
    stint_ages, y_mean, lower, upper, sigmas = deg_model.predict_stint("MEDIUM", max_age=16, base_pace=85.0)
    assert len(y_mean) == 16
    assert y_mean[0] >= 85.0
    assert y_mean[-1] > y_mean[0]
    assert np.all(upper >= y_mean)
    assert np.all(lower <= y_mean)

