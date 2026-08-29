from fastapi import APIRouter
from backend.api.endpoints import sessions, predict, strategy, validate, telemetry, explain, report

api_router = APIRouter()
api_router.include_router(sessions.router)
api_router.include_router(predict.router)
api_router.include_router(strategy.router)
api_router.include_router(validate.router)
api_router.include_router(telemetry.router)
api_router.include_router(explain.router)
api_router.include_router(report.router)
