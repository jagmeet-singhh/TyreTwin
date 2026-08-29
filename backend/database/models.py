import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Index, Text, JSON
)
from sqlalchemy.orm import relationship
from backend.database.connection import Base


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True, index=True)  # e.g., "2024_Bahrain_FP2"
    season = Column(Integer, nullable=False, index=True)
    grand_prix = Column(String(64), nullable=False, index=True)
    session_name = Column(String(32), nullable=False)  # FP1, FP2, FP3, Q, R
    track_name = Column(String(128), nullable=False)
    track_length_km = Column(Float, default=5.412)
    air_temp_c = Column(Float, default=25.0)
    track_temp_c = Column(Float, default=35.0)
    humidity_pct = Column(Float, default=40.0)
    rainfall = Column(Boolean, default=False)
    total_laps = Column(Integer, default=57)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    laps = relationship("LapModel", back_populates="session", cascade="all, delete-orphan")
    stints = relationship("TyreStintModel", back_populates="session", cascade="all, delete-orphan")
    predictions = relationship("PredictionModel", back_populates="session", cascade="all, delete-orphan")


class DriverModel(Base):
    __tablename__ = "drivers"

    code = Column(String(3), primary_key=True, index=True)  # VER, HAM, LEC
    number = Column(Integer, nullable=False)
    full_name = Column(String(64), nullable=False)
    team_name = Column(String(64), nullable=False)
    team_color = Column(String(7), default="#e10600")  # HEX color

    laps = relationship("LapModel", back_populates="driver")
    stints = relationship("TyreStintModel", back_populates="driver")


class LapModel(Base):
    __tablename__ = "laps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False, index=True)
    driver_code = Column(String(3), ForeignKey("drivers.code"), nullable=False, index=True)
    lap_number = Column(Integer, nullable=False, index=True)
    lap_time_sec = Column(Float, nullable=False)
    sector1_sec = Column(Float, nullable=True)
    sector2_sec = Column(Float, nullable=True)
    sector3_sec = Column(Float, nullable=True)
    compound = Column(String(16), nullable=False, index=True)  # SOFT, MEDIUM, HARD, INTERMEDIATE, WET
    tyre_age_lap = Column(Integer, nullable=False)
    stint_number = Column(Integer, default=1)
    
    # Flags & corrections
    is_valid = Column(Boolean, default=True)
    is_pit_in = Column(Boolean, default=False)
    is_pit_out = Column(Boolean, default=False)
    track_status = Column(String(8), default="1")  # 1 = Green, 4 = SC, 6 = VSC, etc.
    fuel_kg = Column(Float, nullable=True)
    
    # ML isolated metrics
    predicted_noise_sec = Column(Float, default=0.0)
    clean_lap_time_sec = Column(Float, nullable=True)
    driver_push_index = Column(Float, default=0.85)
    traffic_flag = Column(Boolean, default=False)
    track_evolution_index = Column(Float, default=0.5)

    session = relationship("SessionModel", back_populates="laps")
    driver = relationship("DriverModel", back_populates="laps")
    telemetry = relationship("TelemetryModel", back_populates="lap", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_session_driver_lap", "session_id", "driver_code", "lap_number"),
    )


class TelemetryModel(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lap_id = Column(Integer, ForeignKey("laps.id"), nullable=False, index=True)
    sample_index = Column(Integer, nullable=False)
    distance_m = Column(Float, nullable=False)
    speed_kmh = Column(Float, nullable=False)
    throttle_pct = Column(Float, nullable=False)
    brake_pct = Column(Float, nullable=False)
    rpm = Column(Integer, nullable=False)
    gear = Column(Integer, nullable=False)
    drs = Column(Integer, default=0)
    time_sec = Column(Float, nullable=False)

    lap = relationship("LapModel", back_populates="telemetry")


class TyreStintModel(Base):
    __tablename__ = "tyre_stints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False, index=True)
    driver_code = Column(String(3), ForeignKey("drivers.code"), nullable=False, index=True)
    stint_number = Column(Integer, nullable=False)
    compound = Column(String(16), nullable=False)
    start_lap = Column(Integer, nullable=False)
    end_lap = Column(Integer, nullable=False)
    total_laps = Column(Integer, nullable=False)
    avg_degradation_sec_per_lap = Column(Float, default=0.06)
    cliff_lap = Column(Integer, nullable=True)

    session = relationship("SessionModel", back_populates="stints")
    driver = relationship("DriverModel", back_populates="stints")


class PredictionModel(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False, index=True)
    driver_code = Column(String(3), nullable=False, index=True)
    compound = Column(String(16), nullable=False)
    model_type = Column(String(32), default="GaussianProcess_XGBoost")
    stint_length = Column(Integer, default=25)
    
    # Outputs
    base_clean_lap_sec = Column(Float, nullable=False)
    degradation_slope = Column(Float, nullable=False)
    cliff_lap = Column(Integer, nullable=False)
    recommended_pit_lap = Column(Integer, nullable=False)
    pit_window_start = Column(Integer, nullable=False)
    pit_window_end = Column(Integer, nullable=False)
    remaining_life_laps = Column(Integer, nullable=False)
    confidence_score = Column(Float, default=0.92)
    
    # Serialized curve: list of dicts with lap, mean_time, lower_ci, upper_ci
    curve_data = Column(JSON, nullable=False)
    shap_attribution = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("SessionModel", back_populates="predictions")


class ValidationResultModel(Base):
    __tablename__ = "validation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    driver_code = Column(String(3), nullable=False)
    compound = Column(String(16), nullable=False)
    
    mae_sec = Column(Float, nullable=False)
    rmse_sec = Column(Float, nullable=False)
    r2_score = Column(Float, nullable=False)
    sample_count = Column(Integer, nullable=False)
    
    comparison_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
