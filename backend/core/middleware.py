# backend/core/middleware.py
from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
import time
import uuid
import json
from typing import Callable, Dict, Any
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta

from ..config.settings import get_settings
from ..config.logging import get_logger, log_api_request, log_api_response, log_security_event
from ..core.security import get_security_headers
from ..core.dependencies import get_redis_client

logger = get_logger(__name__)
settings = get_settings()

# Request ID Middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        
        return response

# Logging Middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests and responses."""
    
    def __init__(self, app, skip_paths: list = None):
        super().__init__(app)
        self.skip_paths = skip_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip logging for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)
        
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        log_api_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            user_id=getattr(request.state, "user_id", None)
        )
        
        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log response
            log_api_response(
                request_id=request_id,
                status_code=response.status_code,
                duration=duration
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request {request_id} failed after {duration:.3f}s: {str(e)}")
            raise

# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        # Add security headers
        security_headers = get_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response

# Rate Limiting Middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis."""
    
    def __init__(self, app, requests_per_minute: int = 60, skip_paths: list = None):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.skip_paths = skip_paths or ["/health", "/metrics"]
        self.redis_client = get_redis_client()
        
        # Fallback in-memory rate limiting if Redis is unavailable
        self.memory_store = defaultdict(deque)
    
    def get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from request state
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fallback to IP address
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    async def check_rate_limit_redis(self, identifier: str) -> bool:
        """Check rate limit using Redis."""
        if not self.redis_client:
            return True
        
        try:
            key = f"rate_limit:{identifier}"
            current_time = int(time.time())
            window_start = current_time - 60  # 1 minute window
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)  # Remove old entries
            pipe.zcard(key)  # Count current entries
            pipe.zadd(key, {str(current_time): current_time})  # Add current request
            pipe.expire(key, 60)  # Set expiration
            
            results = pipe.execute()
            current_requests = results[1]
            
            return current_requests < self.requests_per_minute
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            return True  # Allow request if Redis fails
    
    def check_rate_limit_memory(self, identifier: str) -> bool:
        """Check rate limit using in-memory store."""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Clean old entries
        requests = self.memory_store[identifier]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check if under limit
        if len(requests) >= self.requests_per_minute:
            return False
        
        # Add current request
        requests.append(current_time)
        return True
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)
        
        identifier = self.get_client_identifier(request)
        
        # Check rate limit
        if self.redis_client:
            is_allowed = await self.check_rate_limit_redis(identifier)
        else:
            is_allowed = self.check_rate_limit_memory(identifier)
        
        if not is_allowed:
            log_security_event(
                "RATE_LIMIT_EXCEEDED",
                details=f"Rate limit exceeded for {identifier}",
                ip_address=request.client.host if request.client else None
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed"
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + 60)
                }
            )
        
        return await call_next(request)

# Authentication Middleware
class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Extract user information from JWT tokens."""
    
    def __init__(self, app, skip_paths: list = None):
        super().__init__(app)
        self.skip_paths = skip_paths or [
            "/auth/login", "/auth/register", "/health", "/metrics", 
            "/docs", "/openapi.json", "/redoc"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip authentication for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)
        
        # Try to extract user information from token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                from ..core.security import verify_token
                payload = verify_token(token)
                if payload:
                    request.state.user_id = payload.get("sub")
                    request.state.user_role = payload.get("role")
                    request.state.user_permissions = payload.get("permissions", [])
            except Exception as e:
                logger.debug(f"Token verification failed: {e}")
        
        return await call_next(request)

# Error Handling Middleware
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            return await call_next(request)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unhandled error in {request.method} {request.url.path}: {str(e)}")
            
            # Log security events for suspicious errors
            if "sql" in str(e).lower() or "injection" in str(e).lower():
                log_security_event(
                    "SUSPICIOUS_REQUEST",
                    details=f"Potential SQL injection attempt: {str(e)}",
                    ip_address=request.client.host if request.client else None
                )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            )

# Database Connection Middleware
class DatabaseMiddleware(BaseHTTPMiddleware):
    """Ensure database connections are properly handled."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log database-related errors
            if "database" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Database error: {str(e)}")
            raise

# Request Size Limiting Middleware
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size."""
    
    def __init__(self, app, max_size: int = 50 * 1024 * 1024):  # 50MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Check content length header
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request entity too large",
                    "message": f"Request body must be smaller than {self.max_size} bytes"
                }
            )
        
        return await call_next(request)

# Geolocation Context Middleware
class GeolocationMiddleware(BaseHTTPMiddleware):
    """Add geolocation context to requests for geo-intelligence features."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Extract location from headers (if provided by mobile apps)
        latitude = request.headers.get("X-User-Latitude")
        longitude = request.headers.get("X-User-Longitude")
        
        if latitude and longitude:
            try:
                request.state.user_location = {
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "timestamp": datetime.utcnow()
                }
            except ValueError:
                logger.warning(f"Invalid coordinates in headers: lat={latitude}, lon={longitude}")
        
        return await call_next(request)

# Performance Monitoring Middleware
class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Monitor API performance and identify slow endpoints."""
    
    def __init__(self, app, slow_threshold: float = 2.0):
        super().__init__(app)
        self.slow_threshold = slow_threshold
        self.performance_stats = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Log slow requests
        if duration > self.slow_threshold:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {duration:.3f}s (threshold: {self.slow_threshold}s)"
            )
        
        # Store performance metrics
        endpoint = f"{request.method} {request.url.path}"
        self.performance_stats[endpoint].append(duration)
        
        # Keep only last 100 requests per endpoint
        if len(self.performance_stats[endpoint]) > 100:
            self.performance_stats[endpoint] = self.performance_stats[endpoint][-100:]
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration:.3f}"
        
        return response
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all endpoints."""
        stats = {}
        for endpoint, durations in self.performance_stats.items():
            if durations:
                stats[endpoint] = {
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "count": len(durations)
                }
        return stats

# API Version Middleware
class APIVersionMiddleware(BaseHTTPMiddleware):
    """Handle API versioning through headers."""
    
    def __init__(self, app, default_version: str = "v1"):
        super().__init__(app)
        self.default_version = default_version
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Get API version from header or default
        api_version = request.headers.get("X-API-Version", self.default_version)
        request.state.api_version = api_version
        
        response = await call_next(request)
        
        # Add version to response headers
        response.headers["X-API-Version"] = api_version
        
        return response

# Business Logic Middleware for Sales Intelligence
class SalesIntelligenceMiddleware(BaseHTTPMiddleware):
    """Add sales intelligence context to requests."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Add territory and sales context if user is authenticated
        if hasattr(request.state, "user_id"):
            # This would typically fetch from database
            # For now, we'll just add the structure
            request.state.sales_context = {
                "territory": None,
                "dealer_code": None,
                "division": None,
                "permissions": getattr(request.state, "user_permissions", [])
            }
        
        return await call_next(request)

# Middleware Registration Function
def register_middleware(app):
    """Register all middleware with the FastAPI app."""
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add trusted host middleware for production
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
    
    # Add custom middleware in order (last added is executed first)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(PerformanceMonitoringMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(GeolocationMiddleware)
    app.add_middleware(APIVersionMiddleware)
    app.add_middleware(SalesIntelligenceMiddleware)
    app.add_middleware(DatabaseMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthenticationMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    
    logger.info("All middleware registered successfully")

# Middleware utilities
def get_request_context(request: Request) -> Dict[str, Any]:
    """Extract all context information from request state."""
    context = {
        "request_id": getattr(request.state, "request_id", None),
        "user_id": getattr(request.state, "user_id", None),
        "user_role": getattr(request.state, "user_role", None),
        "user_permissions": getattr(request.state, "user_permissions", []),
        "user_location": getattr(request.state, "user_location", None),
        "api_version": getattr(request.state, "api_version", "v1"),
        "sales_context": getattr(request.state, "sales_context", {}),
    }
    return context

def is_authenticated(request: Request) -> bool:
    """Check if request is authenticated."""
    return hasattr(request.state, "user_id") and request.state.user_id is not None

def has_permission(request: Request, permission: str) -> bool:
    """Check if user has specific permission."""
    if not is_authenticated(request):
        return False
    
    permissions = getattr(request.state, "user_permissions", [])
    return permission in permissions

def get_user_territory(request: Request) -> str:
    """Get user's territory from sales context."""
    sales_context = getattr(request.state, "sales_context", {})
    return sales_context.get("territory")

def get_dealer_code(request: Request) -> str:
    """Get user's dealer code from sales context."""
    sales_context = getattr(request.state, "sales_context", {})
    return sales_context.get("dealer_code")