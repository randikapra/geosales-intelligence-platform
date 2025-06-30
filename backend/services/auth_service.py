from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
import secrets
import redis
from config.settings import settings
from models.user import User
from schemas.auth import TokenData

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password for storing"""
        return self.pwd_context.hash(password)
    
    def authenticate_user(self, db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user = db.query(User).filter(
            (User.username == username) | (User.email == username)
        ).first()
        
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
            
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        # Store refresh token in Redis
        self.redis_client.setex(
            f"refresh_token:{data['sub']}", 
            timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            encoded_jwt
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify and decode JWT token"""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id: str = payload.get("sub")
            token_type_claim: str = payload.get("type")
            
            if user_id is None or token_type_claim != token_type:
                raise credentials_exception
                
            token_data = TokenData(user_id=user_id)
            
        except JWTError:
            raise credentials_exception
            
        return token_data
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token using refresh token"""
        token_data = self.verify_token(refresh_token, "refresh")
        
        # Check if refresh token exists in Redis
        stored_token = self.redis_client.get(f"refresh_token:{token_data.user_id}")
        if stored_token != refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new access token
        access_token = self.create_access_token(data={"sub": token_data.user_id})
        return access_token
    
    def revoke_refresh_token(self, user_id: str):
        """Revoke refresh token by removing from Redis"""
        self.redis_client.delete(f"refresh_token:{user_id}")
    
    def create_password_reset_token(self, user_id: str) -> str:
        """Create password reset token"""
        token = secrets.token_urlsafe(32)
        
        # Store in Redis with 1-hour expiration
        self.redis_client.setex(
            f"password_reset:{token}",
            3600,  # 1 hour
            user_id
        )
        
        return token
    
    def verify_password_reset_token(self, token: str) -> Optional[str]:
        """Verify password reset token and return user_id"""
        user_id = self.redis_client.get(f"password_reset:{token}")
        if user_id:
            # Delete token after use
            self.redis_client.delete(f"password_reset:{token}")
            return user_id
        return None
    
    def change_password(self, db: Session, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            return False
            
        if not self.verify_password(old_password, user.hashed_password):
            return False
        
        user.hashed_password = self.get_password_hash(new_password)
        user.password_changed_at = datetime.utcnow()
        db.commit()
        
        # Revoke all existing refresh tokens for security
        self.revoke_refresh_token(user_id)
        
        return True
    
    def reset_password(self, db: Session, token: str, new_password: str) -> bool:
        """Reset password using reset token"""
        user_id = self.verify_password_reset_token(token)
        
        if not user_id:
            return False
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        
        user.hashed_password = self.get_password_hash(new_password)
        user.password_changed_at = datetime.utcnow()
        db.commit()
        
        # Revoke all existing refresh tokens
        self.revoke_refresh_token(user_id)
        
        return True
    
    def create_api_key(self, user_id: str, name: str, expires_in_days: Optional[int] = None) -> str:
        """Create API key for user"""
        api_key = f"sfa_{secrets.token_urlsafe(32)}"
        
        key_data = {
            "user_id": user_id,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True
        }
        
        if expires_in_days:
            expire_time = datetime.utcnow() + timedelta(days=expires_in_days)
            key_data["expires_at"] = expire_time.isoformat()
            self.redis_client.setex(f"api_key:{api_key}", timedelta(days=expires_in_days), str(key_data))
        else:
            self.redis_client.set(f"api_key:{api_key}", str(key_data))
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated data"""
        key_data = self.redis_client.get(f"api_key:{api_key}")
        if key_data:
            return eval(key_data)  # In production, use json.loads with proper serialization
        return None
    
    def revoke_api_key(self, api_key: str):
        """Revoke API key"""
        self.redis_client.delete(f"api_key:{api_key}")
    
    def get_user_permissions(self, db: Session, user_id: str) -> List[str]:
        """Get user permissions"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return []
        
        permissions = []
        for role in user.roles:
            permissions.extend([perm.name for perm in role.permissions])
        
        return list(set(permissions))  # Remove duplicates
    
    def check_permission(self, db: Session, user_id: str, permission: str) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.get_user_permissions(db, user_id)
        return permission in user_permissions
    
    def log_auth_event(self, user_id: str, event_type: str, ip_address: str, user_agent: str):
        """Log authentication events"""
        event_data = {
            "user_id": user_id,
            "event_type": event_type,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in Redis with 30-day expiration
        event_key = f"auth_log:{user_id}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        self.redis_client.setex(event_key, timedelta(days=30), str(event_data))
    
    def get_auth_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get authentication history for user"""
        pattern = f"auth_log:{user_id}:*"
        keys = self.redis_client.keys(pattern)
        
        events = []
        for key in sorted(keys, reverse=True)[:limit]:
            event_data = self.redis_client.get(key)
            if event_data:
                events.append(eval(event_data))  # In production, use proper JSON handling
        
        return events

# Create singleton instance
auth_service = AuthService()