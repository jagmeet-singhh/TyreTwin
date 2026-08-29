import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark_sessions"
FASTF1_CACHE_DIR = DATA_DIR / "fastf1_cache"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    PROJECT_NAME: str = "TyreIQ"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR}/tyreiq.db"
    )
    
    # Directories
    BENCHMARK_DIR: str = str(BENCHMARK_DIR)
    FASTF1_CACHE_DIR: str = str(FASTF1_CACHE_DIR)
    ENABLE_FASTF1_CACHE: bool = True
    
    # ML Models
    NOISE_MODEL_PATH: str = str(MODELS_DIR / "noise_model.pkl")
    DEGRADATION_MODEL_PATH: str = str(MODELS_DIR / "degradation_model.pkl")
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # F1 Physics & Heuristics Defaults
    FUEL_EFFECT_SEC_PER_10KG: float = 0.30
    INITIAL_FUEL_MASS_KG: float = 105.0
    FUEL_BURN_KG_PER_LAP: float = 1.8
    PIT_LOSS_SEC: float = 22.0
    
    # CORS
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
