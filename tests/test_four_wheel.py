import pytest
from ml.four_wheel_model import FourWheelTyreModel
from ml.train_degradation import TyreDegradationModel


def test_circuit_load_factors():
    # Silverstone must be Front-Left limited
    s_silverstone = FourWheelTyreModel.simulate_four_tyres("2024_Silverstone_Qualifying", "SOFT")
    assert s_silverstone["limiting_corner"] == "FL"
    assert s_silverstone["tyres"]["FL"]["load_factor"] > s_silverstone["tyres"]["FR"]["load_factor"]
    assert s_silverstone["tyres"]["FL"]["is_limiting"] is True

    # Bahrain must be Rear limited (RL or RR)
    s_bahrain = FourWheelTyreModel.simulate_four_tyres("2024_Bahrain_Qualifying", "SOFT")
    assert s_bahrain["limiting_corner"] in ["RL", "RR"]
    assert s_bahrain["tyres"]["RL"]["load_factor"] > s_bahrain["tyres"]["FL"]["load_factor"]

    # Suzuka must be Front limited (FR in esses/degners)
    s_suzuka = FourWheelTyreModel.simulate_four_tyres("2024_Suzuka_Qualifying", "SOFT")
    assert s_suzuka["limiting_corner"] == "FR"


def test_four_wheel_curves_and_temperatures():
    res = FourWheelTyreModel.simulate_four_tyres("2024_Bahrain_Practice", "MEDIUM", stint_length=20)
    tyres = res["tyres"]
    
    for corner in ["FL", "FR", "RL", "RR"]:
        assert corner in tyres
        t = tyres[corner]
        assert 0.0 <= t["remaining_wear_pct"] <= 100.0
        assert t["cliff_lap"] > 5
        assert t["surface_temp_c"] > 30.0
        assert t["carcass_temp_c"] > 30.0
        assert t["thermal_status"] in ["Optimal Window", "Overheating", "Thermal Blistering", "Graining Risk"]

    # Verify 20 laps of curves exist for each tyre
    assert len(res["curves"]["FL"]) == 20
    assert len(res["history"]) == 20
    assert res["curves"]["FL"][-1]["remaining_wear_pct"] < res["curves"]["FL"][0]["remaining_wear_pct"]


def test_four_corner_gp_models():
    deg_model = TyreDegradationModel.load()
    assert len(deg_model.gp_corner_models) >= 12
    
    # Predict for each corner and verify that load factors produce distinct pace curves
    ages_fl, pace_fl, lower_fl, upper_fl, sig_fl = deg_model.predict_corner_stint("SOFT", "FL", max_age=15, base_pace=90.0, load_factor=1.28)
    ages_rr, pace_rr, lower_rr, upper_rr, sig_rr = deg_model.predict_corner_stint("SOFT", "RR", max_age=15, base_pace=90.0, load_factor=0.90)
    
    assert len(pace_fl) == 15
    assert len(pace_rr) == 15
    # FL with higher load factor degrades more than RR
    assert pace_fl[-1] >= pace_rr[-1]


def test_pdf_report_all_four_tyres():
    from app.utils.pdf_generator import PDFReportGenerator
    payload = {
        "session_id": "2024_Silverstone_Qualifying",
        "driver_code": "VER",
        "compound": "SOFT",
        "four_tyres": FourWheelTyreModel.simulate_four_tyres("2024_Silverstone_Qualifying", "SOFT")
    }
    pdf_bytes = PDFReportGenerator.create_pitwall_report(payload)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF-")
