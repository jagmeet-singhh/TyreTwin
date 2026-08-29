from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.database.models import (
    SessionModel, DriverModel, LapModel, TelemetryModel,
    TyreStintModel, PredictionModel, ValidationResultModel
)


def get_session(db: Session, session_id: str) -> Optional[SessionModel]:
    return db.query(SessionModel).filter(SessionModel.id == session_id).first()


def list_sessions(db: Session, season: Optional[int] = None) -> List[SessionModel]:
    query = db.query(SessionModel)
    if season:
        query = query.filter(SessionModel.season == season)
    return query.order_by(SessionModel.season.desc(), SessionModel.grand_prix).all()


def create_or_update_session(db: Session, session_data: Dict[str, Any]) -> SessionModel:
    session = db.query(SessionModel).filter(SessionModel.id == session_data["id"]).first()
    if not session:
        session = SessionModel(**session_data)
        db.add(session)
    else:
        for key, val in session_data.items():
            setattr(session, key, val)
    db.commit()
    db.refresh(session)
    return session


def create_or_update_driver(db: Session, driver_data: Dict[str, Any]) -> DriverModel:
    driver = db.query(DriverModel).filter(DriverModel.code == driver_data["code"]).first()
    if not driver:
        driver = DriverModel(**driver_data)
        db.add(driver)
    else:
        for key, val in driver_data.items():
            setattr(driver, key, val)
    db.commit()
    db.refresh(driver)
    return driver


def list_drivers(db: Session) -> List[DriverModel]:
    return db.query(DriverModel).order_by(DriverModel.team_name, DriverModel.code).all()


def save_laps_batch(db: Session, laps_data: List[Dict[str, Any]]):
    db.bulk_insert_mappings(LapModel, laps_data)
    db.commit()


def get_driver_laps(
    db: Session, session_id: str, driver_code: str, compound: Optional[str] = None
) -> List[LapModel]:
    query = db.query(LapModel).filter(
        LapModel.session_id == session_id,
        LapModel.driver_code == driver_code,
        LapModel.is_valid == True
    )
    if compound:
        query = query.filter(LapModel.compound == compound.upper())
    return query.order_by(LapModel.lap_number).all()


def save_telemetry_batch(db: Session, telemetry_data: List[Dict[str, Any]]):
    db.bulk_insert_mappings(TelemetryModel, telemetry_data)
    db.commit()


def get_lap_telemetry(db: Session, lap_id: int) -> List[TelemetryModel]:
    return db.query(TelemetryModel).filter(
        TelemetryModel.lap_id == lap_id
    ).order_by(TelemetryModel.sample_index).all()


def save_prediction(db: Session, prediction_data: Dict[str, Any]) -> PredictionModel:
    pred = PredictionModel(**prediction_data)
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def get_latest_prediction(
    db: Session, session_id: str, driver_code: str, compound: str
) -> Optional[PredictionModel]:
    return db.query(PredictionModel).filter(
        PredictionModel.session_id == session_id,
        PredictionModel.driver_code == driver_code,
        PredictionModel.compound == compound.upper()
    ).order_by(PredictionModel.created_at.desc()).first()


def save_validation_result(db: Session, val_data: Dict[str, Any]) -> ValidationResultModel:
    val = ValidationResultModel(**val_data)
    db.add(val)
    db.commit()
    db.refresh(val)
    return val
