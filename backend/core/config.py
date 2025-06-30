# backend/core/config.py
"""
Configuration settings for the application
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    SECRET_KEY: str = "your-super-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://geosales_user:geosales_pass@localhost:5432/geosales_db"
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = "geosales_db"
    DATABASE_USER: str = "geosales_user"
    DATABASE_PASSWORD: str = "geosales_pass"
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Kafka Configuration
    KAFKA_BROKER: str = "localhost:9092"
    KAFKA_TOPIC_GPS: str = "gps_data"
    KAFKA_TOPIC_SALES: str = "sales_data"
    KAFKA_TOPIC_ALERTS: str = "alerts"
    
    # External APIs
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    MAPBOX_ACCESS_TOKEN: Optional[str] = None
    OPENWEATHER_API_KEY: Optional[str] = None
    
    # ML Configuration
    ML_MODEL_PATH: str = "/app/data/models"
    ML_BATCH_SIZE: int = 32
    ML_MAX_WORKERS: int = 4
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    
    # Security
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    JWT_SECRET_KEY: str = "your-jwt-secret-key"
    
    # Feature Flags
    ENABLE_REAL_TIME_TRACKING: bool = True
    ENABLE_PREDICTIVE_ANALYTICS: bool = True
    ENABLE_ROUTE_OPTIMIZATION: bool = True
    ENABLE_MOBILE_API: bool = True
    
    # File Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET: str = "geosales-data"
    
    # Monitoring
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3001
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

