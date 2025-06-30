# backend/config/settings.py
from pydantic import BaseSettings
from typing import Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "GeoSales Intelligence Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database Settings
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/geosales_db"
    DATABASE_URL_ASYNC: str = "postgresql+asyncpg://user:password@localhost:5432/geosales_db"
    TEST_DATABASE_URL: str = "postgresql://user:password@localhost:5432/geosales_test_db"
    
    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # JWT Settings
    JWT_SECRET_KEY: str = "jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRES: int = 1800  # 30 minutes
    JWT_REFRESH_TOKEN_EXPIRES: int = 604800  # 7 days
    
    # OAuth Settings
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    
    # Email Settings
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    # Machine Learning Settings
    ML_MODEL_PATH: str = "data/models/trained_models"
    ML_BATCH_SIZE: int = 32
    ML_PREDICTION_CACHE_TTL: int = 1800  # 30 minutes
    
    # Model Paths
    SALES_FORECASTING_MODEL: str = "data/models/trained_models/sales_forecaster.pkl"
    CUSTOMER_SEGMENTATION_MODEL: str = "data/models/trained_models/customer_segmentation.pkl"
    ROUTE_OPTIMIZATION_MODEL: str = "data/models/trained_models/route_optimizer.pkl"
    CHURN_PREDICTION_MODEL: str = "data/models/trained_models/churn_predictor.pkl"
    
    # API Keys
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    MAPBOX_API_KEY: Optional[str] = None
    WEATHER_API_KEY: Optional[str] = None
    
    # Pagination Settings
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    
    # CORS Settings
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: list = [".csv", ".xlsx", ".xls", ".json"]
    UPLOAD_DIRECTORY: str = "data/uploads"
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/app.log"
    
    # Monitoring Settings
    PROMETHEUS_ENABLED: bool = True
    HEALTH_CHECK_INTERVAL: int = 30
    
    # Background Tasks
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Streaming Settings
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_GPS_TOPIC: str = "gps_data"
    KAFKA_SALES_TOPIC: str = "sales_data"
    
    # Geographical Settings
    DEFAULT_COUNTRY: str = "LK"
    DEFAULT_TIMEZONE: str = "Asia/Colombo"
    GEO_SEARCH_RADIUS_KM: float = 50.0
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Environment-specific settings
class DevelopmentSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

class ProductionSettings(Settings):
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"

class TestingSettings(Settings):
    DATABASE_URL: str = "sqlite:///./test.db"
    REDIS_URL: str = "redis://localhost:6379/1"
    JWT_ACCESS_TOKEN_EXPIRES: int = 300  # 5 minutes for testing

def get_settings_by_env(env: str = "development") -> Settings:
    if env == "development":
        return DevelopmentSettings()
    elif env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return Settings()