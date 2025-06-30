# backend/main.py
"""
GeoSales Intelligence Platform - Main API Entry Point
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from core.config import settings
from core.database import engine, Base
from core.middleware import LoggingMiddleware, RateLimitMiddleware
from core.exceptions import custom_exception_handler
from api.v1.endpoints import (
    auth, customers, dealers, sales, 
    analytics, routes, predictions, reports
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting GeoSales Intelligence Platform...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Initialize ML models
    from ml_engine.deployment.model_registry import ModelRegistry
    model_registry = ModelRegistry()
    await model_registry.load_models()
    
    yield
    
    # Shutdown
    logger.info("Shutting down GeoSales Intelligence Platform...")


# Create FastAPI application
app = FastAPI(
    title="GeoSales Intelligence Platform API",
    description="AI-powered geospatial sales intelligence and route optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Add exception handlers
app.add_exception_handler(HTTPException, custom_exception_handler)

# Include routers
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)
app.include_router(
    customers.router,
    prefix="/api/v1/customers",
    tags=["Customers"]
)
app.include_router(
    dealers.router,
    prefix="/api/v1/dealers",
    tags=["Dealers"]
)
app.include_router(
    sales.router,
    prefix="/api/v1/sales",
    tags=["Sales"]
)
app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["Analytics"]
)
app.include_router(
    routes.router,
    prefix="/api/v1/routes",
    tags=["Route Optimization"]
)
app.include_router(
    predictions.router,
    prefix="/api/v1/predictions",
    tags=["Predictions"]
)
app.include_router(
    reports.router,
    prefix="/api/v1/reports",
    tags=["Reports"]
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GeoSales Intelligence Platform API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2025-06-22T00:00:00Z",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4
    )



