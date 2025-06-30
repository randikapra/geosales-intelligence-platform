# backend/core/dependencies.py
from fastapi import Depends, HTTPException, status, Query, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List, Generator
import redis
from functools import lru_cache
import jwt
import uuid
from datetime import datetime

from ..config.database import get_db
from ..config.settings import get_settings
from ..models.user import User
from ..core.security import verify_token, get_current_user_from_token
from ..config.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()
security = HTTPBearer()

# Redis client for caching
@lru_cache()
def get_redis_client():
    """Get Redis client instance."""
    try:
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return None

# Database session dependency
def get_database() -> Generator[Session, None, None]:
    """Database session dependency."""
    return get_db()

# Authentication dependencies
def get_current_user(
    db: Session = Depends(get_database),
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user."""
    try:
        token = credentials.credentials
        payload = verify_token(token)
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled"
            )
        
        return user
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

# Optional authentication
def get_current_user_optional(
    db: Session = Depends(get_database),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    if not credentials:
        return None
    
    try:
        return get_current_user(db, credentials)
    except HTTPException:
        return None

# Admin user dependency
def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Ensure current user is an admin."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Super user dependency
def get_current_super_user(current_user: User = Depends(get_current_user)) -> User:
    """Ensure current user is a super user."""
    if not current_user.is_super_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super user access required"
        )
    return current_user

# Active user dependency
def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Ensure current user is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

# Pagination dependency
class PaginationParams:
    def __init__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(20, ge=1, le=100, description="Page size"),
        sort_by: Optional[str] = Query(None, description="Sort field"),
        sort_order: str = Query("asc", regex="^(asc|desc)$", description="Sort order")
    ):
        self.page = page
        self.size = min(size, settings.MAX_PAGE_SIZE)
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.offset = (page - 1) * self.size

def get_pagination_params() -> PaginationParams:
    """Get pagination parameters."""
    return PaginationParams()

# Search and filtering dependencies
class SearchParams:
    def __init__(
        self,
        q: Optional[str] = Query(None, description="Search query"),
        fields: Optional[List[str]] = Query(None, description="Fields to search"),
        filters: Optional[Dict[str, Any]] = Query(None, description="Additional filters")
    ):
        self.query = q
        self.fields = fields or []
        self.filters = filters or {}

def get_search_params() -> SearchParams:
    """Get search parameters."""
    return SearchParams()

# Geographic filtering dependency
class GeoParams:
    def __init__(
        self,
        lat: Optional[float] = Query(None, description="Latitude"),
        lng: Optional[float] = Query(None, description="Longitude"),
        radius: Optional[float] = Query(None, description="Search radius in km"),
        bounds: Optional[str] = Query(None, description="Bounding box coordinates")
    ):
        self.latitude = lat
        self.longitude = lng
        self.radius = radius or settings.GEO_SEARCH_RADIUS_KM
        self.bounds = bounds

def get_geo_params() -> GeoParams:
    """Get geographic parameters."""
    return GeoParams()

# Date range filtering dependency
class DateRangeParams:
    def __init__(
        self,
        start_date: Optional[datetime] = Query(None, description="Start date"),
        end_date: Optional[datetime] = Query(None, description="End date"),
        date_field: str = Query("created_at", description="Date field to filter on")
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.date_field = date_field

def get_date_range_params() -> DateRangeParams:
    """Get date range parameters."""
    return DateRangeParams()

# Request context dependency
class RequestContext:
    def __init__(self, request: Request):
        self.request = request
        self.request_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.ip_address = self.get_client_ip()
        self.user_agent = request.headers.get("user-agent", "")
        
    def get_client_ip(self) -> str:
        """Get client IP address."""
        forwarded = self.request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return self.request.client.host if self.request.client else "unknown"

def get_request_context(request: Request) -> RequestContext:
    """Get request context."""
    return RequestContext(request)

# Cache dependency
class CacheService:
    def __init__(self):
        self.redis_client = get_redis_client()
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if not self.redis_client:
            return None
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set value in cache."""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.setex(key, ttl or settings.REDIS_CACHE_TTL, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.delete(key) > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

def get_cache_service() -> CacheService:
    """Get cache service."""
    return CacheService()

# Rate limiting dependency
class RateLimiter:
    def __init__(self):
        self.redis_client = get_redis_client()
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed based on rate limit."""
        if not self.redis_client:
            return True  # Allow if Redis is not available
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = pipe.execute()
            
            current_count = results[0]
            return current_count <= limit
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True  # Allow on error

def get_rate_limiter() -> RateLimiter:
    """Get rate limiter."""
    return RateLimiter()

# Database transaction dependency
class TransactionManager:
    def __init__(self, db: Session):
        self.db = db
    
    def __enter__(self):
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.db.rollback()
        else:
            self.db.commit()

def get_transaction_manager(db: Session = Depends(get_database)) -> TransactionManager:
    """Get transaction manager."""
    return TransactionManager(db)

# Permission checking dependency
def check_permission(permission: str):
    """Check if user has specific permission."""
    def permission_checker(current_user: User = Depends(get_current_user)):
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker

# Role checking dependency
def require_role(role: str):
    """Require specific user role."""
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return current_user
    return role_checker

# API key dependency
def get_api_key(api_key: str = Query(..., description="API Key")):
    """Validate API key."""
    # This would typically check against a database of valid API keys
    valid_api_keys = ["your-api-key-here"]  # Replace with actual validation
    
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

# File upload dependency
class FileUploadParams:
    def __init__(
        self,
        max_size: int = settings.MAX_FILE_SIZE,
        allowed_types: List[str] = settings.ALLOWED_FILE_TYPES
    ):
        self.max_size = max_size
        self.allowed_types = allowed_types

def get_file_upload_params() -> FileUploadParams:
    """Get file upload parameters."""
    return FileUploadParams()

# Export all dependencies
__all__ = [
    "get_database",
    "get_redis_client",
    "get_current_user",
    "get_current_user_optional",
    "get_current_admin_user",
    "get_current_super_user",
    "get_current_active_user",
    "get_pagination_params",
    "get_search_params",
    "get_geo_params",
    "get_date_range_params",
    "get_request_context",
    "get_cache_service",
    "get_rate_limiter",
    "get_transaction_manager",
    "check_permission",
    "require_role",
    "get_api_key",
    "get_file_upload_params",
    "PaginationParams",
    "SearchParams",
    "GeoParams",
    "DateRangeParams",
    "RequestContext",
    "CacheService",
    "RateLimiter",
    "TransactionManager",
    "FileUploadParams"
]