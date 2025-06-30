"""
Real-time Prediction API for SFA ML Models
Provides high-performance, low-latency predictions with caching, monitoring, and scaling capabilities
"""
import os
import sys
import logging
import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import redis
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
from contextlib import asynccontextmanager
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import predictor
from predictor import UniversalPredictor, create_predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Real-time prediction request model"""
    model_name: str = Field(..., description="Name of the ML model to use")
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Input data for prediction")
    data_type: str = Field(default="mixed", description="Type of data (sales, customer, gps, mixed)")
    use_cache: bool = Field(default=True, description="Whether to use caching")
    return_probabilities: bool = Field(default=False, description="Return prediction probabilities")
    return_confidence: bool = Field(default=True, description="Return confidence scores")
    return_explanation: bool = Field(default=False, description="Return prediction explanations")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        allowed_models = [
            'sales_forecaster', 'demand_forecaster', 'customer_segmentation',
            'churn_prediction', 'dealer_performance', 'route_optimizer',
            'territory_optimizer'
        ]
        if v not in allowed_models:
            raise ValueError(f'Model name must be one of: {allowed_models}')
        return v

class PredictionResponse(BaseModel):
    """Real-time prediction response model"""
    predictions: Union[List[Any], Any]
    probabilities: Optional[List[List[float]]] = None
    confidence_scores: Optional[List[float]] = None
    explanations: Optional[List[Dict[str, Any]]] = None
    model_name: str
    data_type: str
    prediction_time: float
    timestamp: datetime
    cached: bool = False
    request_id: str

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    uptime: float
    total_predictions: int
    cache_hit_rate: float
    avg_response_time: float
    models_loaded: List[str]
    system_metrics: Dict[str, Any]

@dataclass
class PredictionMetrics:
    """Metrics tracking for predictions"""
    request_id: str
    model_name: str
    data_type: str
    response_time: float
    cached: bool
    timestamp: datetime
    success: bool
    error_message: str = None

class MetricsCollector:
    """Collects and manages prediction metrics"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.model_stats = defaultdict(lambda: defaultdict(list))
        self.lock = threading.Lock()
    
    def record_prediction(self, metrics: PredictionMetrics):
        """Record prediction metrics"""
        with self.lock:
            self.metrics_history.append(metrics)
            self.model_stats[metrics.model_name]['response_times'].append(metrics.response_time)
            self.model_stats[metrics.model_name]['cached'].append(metrics.cached)
            self.model_stats[metrics.model_name]['success'].append(metrics.success)
    
    def get_cache_hit_rate(self, time_window: int = 3600) -> float:
        """Get cache hit rate for the last time window (seconds)"""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return 0.0
        
        cached_count = sum(1 for m in recent_metrics if m.cached)
        return cached_count / len(recent_metrics)
    
    def get_avg_response_time(self, time_window: int = 3600) -> float:
        """Get average response time for the last time window (seconds)"""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return 0.0
        
        response_times = [m.response_time for m in recent_metrics]
        return sum(response_times) / len(response_times)
    
    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get statistics for a specific model"""
        if model_name not in self.model_stats:
            return {}
        
        stats = self.model_stats[model_name]
        return {
            'total_predictions': len(stats['response_times']),
            'avg_response_time': np.mean(stats['response_times']) if stats['response_times'] else 0,
            'cache_hit_rate': np.mean(stats['cached']) if stats['cached'] else 0,
            'success_rate': np.mean(stats['success']) if stats['success'] else 0,
            'p95_response_time': np.percentile(stats['response_times'], 95) if stats['response_times'] else 0,
            'p99_response_time': np.percentile(stats['response_times'], 99) if stats['response_times'] else 0
        }

class CacheManager:
    """Manages prediction caching with Redis"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()  # Test connection
            self.cache_enabled = True
            logger.info("Redis cache enabled")
        except Exception as e:
            logger.warning(f"Redis not available, caching disabled: {e}")
            self.cache_enabled = False
            self.redis_client = None
        
        self.default_ttl = default_ttl
        self.local_cache = {}  # Fallback local cache
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def _generate_cache_key(self, model_name: str, data: Any, data_type: str) -> str:
        """Generate cache key from input data"""
        # Convert data to string and hash it
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        return f"sfa_pred:{model_name}:{data_type}:{data_hash}"
    
    def get(self, model_name: str, data: Any, data_type: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction"""
        if not self.cache_enabled:
            return None
        
        cache_key = self._generate_cache_key(model_name, data, data_type)
        
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.cache_stats['hits'] += 1
                    return pickle.loads(cached_data)
            else:
                # Fallback to local cache
                if cache_key in self.local_cache:
                    self.cache_stats['hits'] += 1
                    return self.local_cache[cache_key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, model_name: str, data: Any, data_type: str, 
            prediction: Dict[str, Any], ttl: int = None):
        """Cache prediction result"""
        if not self.cache_enabled:
            return
        
        cache_key = self._generate_cache_key(model_name, data, data_type)
        ttl = ttl or self.default_ttl
        
        try:
            if self.redis_client:
                self.redis_client.setex(cache_key, ttl, pickle.dumps(prediction))
            else:
                # Fallback to local cache (no TTL for simplicity)
                self.local_cache[cache_key] = prediction
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def clear(self, pattern: str = "sfa_pred:*"):
        """Clear cache entries matching pattern"""
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} cache entries")
            else:
                # Clear local cache
                keys_to_remove = [k for k in self.local_cache.keys() if k.startswith("sfa_pred:")]
                for key in keys_to_remove:
                    del self.local_cache[key]
                logger.info(f"Cleared {len(keys_to_remove)} local cache entries")
                
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'enabled': self.cache_enabled,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

class RealTimePredictor:
    """High-performance real-time prediction engine"""
    
    def __init__(self, 
                 predictor: UniversalPredictor = None,
                 cache_manager: CacheManager = None,
                 metrics_collector: MetricsCollector = None,
                 max_concurrent_requests: int = 100):
        
        self.predictor = predictor or create_predictor()
        self.cache_manager = cache_manager or CacheManager()
        self.metrics_collector = metrics_collector or MetricsCollector()
        
        # Request throttling
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_count = 0
        self.start_time = time.time()
        
        # Model warm-up
        self._warm_up_models()
        
        logger.info("Real-time predictor initialized")
    
    def _warm_up_models(self):
        """Warm up models by making dummy predictions"""
        logger.info("Warming up models...")
        
        # Create dummy data for each data type
        dummy_data = {
            'sales': pd.DataFrame({
                'DistributorCode': ['DIST001'],
                'UserCode': ['USER001'],
                'FinalValue': [1000.0],
                'Date': [datetime.now()]
            }),
            'customer': pd.DataFrame({
                'No.': ['CUST001'],
                'City': ['COLOMBO'],
                'Contact': ['Contact1']
            }),
            'gps': pd.DataFrame({
                'DivisionCode': ['202'],
                'UserCode': ['AT-001651'],
                'Latitude': [9.708725],
                'Longitude': [80.074470],
                'RecievedDate': [datetime.now()]
            })
        }
        
        # Warm up each model with appropriate data
        model_data_map = {
            'sales_forecaster': 'sales',
            'demand_forecaster': 'sales',
            'customer_segmentation': 'customer',
            'churn_prediction': 'customer',
            'dealer_performance': 'sales',
            'route_optimizer': 'gps',
            'territory_optimizer': 'gps'
        }
        
        for model_name, data_type in model_data_map.items():
            try:
                self.predictor.predict(model_name, dummy_data[data_type], data_type)
                logger.info(f"Warmed up {model_name}")
            except Exception as e:
                logger.warning(f"Failed to warm up {model_name}: {e}")
        
        logger.info("Model warm-up completed")
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make real-time prediction"""
        request_id = f"req_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
        start_time = time.time()
        
        async with self.semaphore:
            try:
                # Check cache first
                cached_result = None
                if request.use_cache:
                    cached_result = self.cache_manager.get(
                        request.model_name, request.data, request.data_type
                    )
                
                if cached_result:
                    # Return cached result
                    response_time = time.time() - start_time
                    
                    # Record metrics
                    metrics = PredictionMetrics(
                        request_id=request_id,
                        model_name=request.model_name,
                        data_type=request.data_type,
                        response_time=response_time,
                        cached=True,
                        timestamp=datetime.now(),
                        success=True
                    )
                    self.metrics_collector.record_prediction(metrics)
                    
                    return PredictionResponse(
                        predictions=cached_result['predictions'],
                        probabilities=cached_result.get('probabilities') if request.return_probabilities else None,
                        confidence_scores=cached_result.get('confidence_scores') if request.return_confidence else None,
                        explanations=cached_result.get('explanations') if request.return_explanation else None,
                        model_name=request.model_name,
                        data_type=request.data_type,
                        prediction_time=response_time,
                        timestamp=datetime.now(),
                        cached=True,
                        request_id=request_id
                    )
                
                # Convert data to DataFrame if needed
                if isinstance(request.data, dict):
                    input_data = pd.DataFrame([request.data])
                elif isinstance(request.data, list):
                    input_data = pd.DataFrame(request.data)
                else:
                    input_data = request.data
                
                # Make prediction
                prediction_result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.predictor.predict,
                    request.model_name,
                    input_data,
                    request.data_type
                )
                
                # Cache the result
                if request.use_cache:
                    self.cache_manager.set(
                        request.model_name,
                        request.data,
                        request.data_type,
                        prediction_result
                    )
                
                response_time = time.time() - start_time
                
                # Record metrics
                metrics = PredictionMetrics(
                    request_id=request_id,
                    model_name=request.model_name,
                    data_type=request.data_type,
                    response_time=response_time,
                    cached=False,
                    timestamp=datetime.now(),
                    success=True
                )
                self.metrics_collector.record_prediction(metrics)
                
                return PredictionResponse(
                    predictions=prediction_result['predictions'],
                    probabilities=prediction_result.get('probabilities') if request.return_probabilities else None,
                    confidence_scores=prediction_result.get('confidence_scores') if request.return_confidence else None,
                    explanations=prediction_result.get('explanations') if request.return_explanation else None,
                    model_name=request.model_name,
                    data_type=request.data_type,
                    prediction_time=response_time,
                    timestamp=datetime.now(),
                    cached=False,
                    request_id=request_id
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                error_message = str(e)
                
                # Record error metrics
                metrics = PredictionMetrics(
                    request_id=request_id,
                    model_name=request.model_name,
                    data_type=request.data_type,
                    response_time=response_time,
                    cached=False,
                    timestamp=datetime.now(),
                    success=False,
                    error_message=error_message
                )
                self.metrics_collector.record_prediction(metrics)
                
                logger.error(f"Prediction failed for {request_id}: {error_message}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {error_message}")
    
    def get_health_status(self) -> HealthResponse:
        """Get health status and system metrics"""
        uptime = time.time() - self.start_time
        total_predictions = len(self.metrics_collector.metrics_history)
        cache_hit_rate = self.metrics_collector.get_cache_hit_rate()
        avg_response_time = self.metrics_collector.get_avg_response_time()
        
        # Get system metrics
        import psutil
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
        
        # Get loaded models
        loaded_models = list(self.predictor.models.keys()) if hasattr(self.predictor, 'models') else []
        
        return HealthResponse(
            status="healthy",
            uptime=uptime,
            total_predictions=total_predictions,
            cache_hit_rate=cache_hit_rate,
            avg_response_time=avg_response_time,
            models_loaded=loaded_models,
            system_metrics=system_metrics
        )

# Global instances
rt_predictor = None
app = FastAPI(
    title="SFA Real-time ML Prediction API",
    description="High-performance real-time predictions for SFA ML models",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global rt_predictor
    
    # Startup
    logger.info("Starting SFA Real-time Prediction API")
    rt_predictor = RealTimePredictor()
    yield
    
    # Shutdown
    logger.info("Shutting down SFA Real-time Prediction API")

app.router.lifespan_context = lifespan

# API endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """Make real-time prediction"""
    return await rt_predictor.predict(request)

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch_endpoint(requests: List[PredictionRequest]):
    """Make multiple predictions in parallel"""
    tasks = [rt_predictor.predict(req) for req in requests]
    return await asyncio.gather(*tasks)

@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check endpoint"""
    return rt_predictor.get_health_status()

@app.get("/metrics")
async def metrics_endpoint():
    """Get prediction metrics"""
    return {
        'cache_stats': rt_predictor.cache_manager.get_stats(),
        'recent_predictions': len(rt_predictor.metrics_collector.metrics_history),
        'cache_hit_rate': rt_predictor.metrics_collector.get_cache_hit_rate(),
        'avg_response_time': rt_predictor.metrics_collector.get_avg_response_time(),
        'model_stats': {
            model: rt_predictor.metrics_collector.get_model_stats(model)
            for model in ['sales_forecaster', 'demand_forecaster', 'customer_segmentation',
                         'churn_prediction', 'dealer_performance', 'route_optimizer',
                         'territory_optimizer']
        }
    }

@app.post("/cache/clear")
async def clear_cache_endpoint():
    """Clear prediction cache"""
    rt_predictor.cache_manager.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/models")
async def list_models_endpoint():
    """List available models"""
    return {
        'available_models': [
            'sales_forecaster', 'demand_forecaster', 'customer_segmentation',
            'churn_prediction', 'dealer_performance', 'route_optimizer',
            'territory_optimizer'
        ],
        'data_types': ['sales', 'customer', 'gps', 'mixed']
    }

# SFA-specific endpoints
@app.post("/predict/sales-forecast")
async def predict_sales_forecast(
    distributor_code: str,
    user_code: str,
    historical_data: List[Dict[str, Any]],
    forecast_horizon: int = 30
):
    """Predict sales forecast for specific distributor and user"""
    request = PredictionRequest(
        model_name="sales_forecaster",
        data=historical_data,
        data_type="sales"
    )
    return await rt_predictor.predict(request)

@app.post("/predict/customer-churn")
async def predict_customer_churn(customer_data: Dict[str, Any]):
    """Predict customer churn probability"""
    request = PredictionRequest(
        model_name="churn_prediction",
        data=customer_data,
        data_type="customer",
        return_probabilities=True
    )
    return await rt_predictor.predict(request)

@app.post("/predict/route-optimization")
async def predict_route_optimization(
    gps_data: List[Dict[str, Any]],
    user_code: str,
    tour_code: str
):
    """Optimize route based on GPS data"""
    request = PredictionRequest(
        model_name="route_optimizer",
        data=gps_data,
        data_type="gps"
    )
    return await rt_predictor.predict(request)

@app.post("/predict/dealer-performance")
async def predict_dealer_performance(
    dealer_data: Dict[str, Any],
    sales_data: List[Dict[str, Any]]
):
    """Predict dealer performance classification"""
    # Combine dealer and sales data
    combined_data = {**dealer_data, 'sales_history': sales_data}
    
    request = PredictionRequest(
        model_name="dealer_performance",
        data=combined_data,
        data_type="mixed"
    )
    return await rt_predictor.predict(request)

# WebSocket endpoint for real-time streaming predictions
@app.websocket("/ws/predictions")
async def websocket_predictions(websocket):
    """WebSocket endpoint for streaming predictions"""
    await websocket.accept()
    
    try:
        while True:
            # Receive prediction request
            data = await websocket.receive_json()
            
            try:
                request = PredictionRequest(**data)
                response = await rt_predictor.predict(request)
                
                # Send prediction response
                await websocket.send_json(response.dict())
                
            except Exception as e:
                await websocket.send_json({
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Background tasks
@app.on_event("startup")
async def startup_tasks():
    """Background tasks to run on startup"""
    # Schedule periodic cache cleanup
    async def cleanup_cache():
        while True:
            await asyncio.sleep(3600)  # Run every hour
            try:
                # Clear old cache entries
                rt_predictor.cache_manager.clear("sfa_pred:*")
                logger.info("Periodic cache cleanup completed")
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    # Schedule periodic metrics cleanup
    async def cleanup_metrics():
        while True:
            await asyncio.sleep(3600)  # Run every hour
            try:
                # Keep only recent metrics
                cutoff_time = datetime.now() - timedelta(hours=24)
                old_metrics = [
                    m for m in rt_predictor.metrics_collector.metrics_history 
                    if m.timestamp < cutoff_time
                ]
                for _ in range(len(old_metrics)):
                    rt_predictor.metrics_collector.metrics_history.popleft()
                
                logger.info(f"Cleaned up {len(old_metrics)} old metrics")
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
    
    # Start background tasks
    asyncio.create_task(cleanup_cache())
    asyncio.create_task(cleanup_metrics())

# Production server configuration
def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the production server"""
    uvicorn.run(
        "real_time_predictor:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True,
        log_level="info"
    )

# CLI interface
def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SFA Real-time Prediction API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--redis-url', default='redis://localhost:6379', help='Redis URL for caching')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Run in development mode
        uvicorn.run(
            "real_time_predictor:app",
            host=args.host,
            port=args.port,
            reload=True,
            debug=True
        )
    else:
        # Run in production mode
        run_server(args.host, args.port, args.workers)

if __name__ == "__main__":
    main()