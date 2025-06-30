# backend/core/security.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib
import hmac
from email_validator import validate_email, EmailNotValidError

from ..config.settings import get_settings
from ..config.logging import get_logger, log_security_event

logger = get_logger(__name__)
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security constants
ALGORITHM = settings.JWT_ALGORITHM
SECRET_KEY = settings.JWT_SECRET_KEY
ACCESS_TOKEN_EXPIRE_MINUTES = settings.JWT_ACCESS_TOKEN_EXPIRES
REFRESH_TOKEN_EXPIRE_DAYS = settings.JWT_REFRESH_TOKEN_EXPIRES

# Password security
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def validate_password_strength(password: str) -> Dict[str, Union[bool, str]]:
    """Validate password strength."""
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        errors.append("Password must contain at least one special character")
    
    # Check for common passwords
    common_passwords = [
        "password", "123456", "password123", "admin", "qwerty",
        "letmein", "welcome", "monkey", "dragon", "master"
    ]
    
    if password.lower() in common_passwords:
        errors.append("Password is too common")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }

# JWT Token handling
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(seconds=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Token creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create access token"
        )

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Refresh token creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create refresh token"
        )

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        log_security_event("TOKEN_EXPIRED", details="JWT token has expired")
        return None
    except jwt.InvalidTokenError as e:
        log_security_event("INVALID_TOKEN", details=f"Invalid JWT token: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None

def get_current_user_from_token(token: str) -> Optional[str]:
    """Extract user ID from token."""
    payload = verify_token(token)
    if payload and payload.get("type") == "access":
        return payload.get("sub")
    return None

def verify_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify refresh token."""
    payload = verify_token(token)
    if payload and payload.get("type") == "refresh":
        return payload
    return None

# API Key generation and validation
def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    """Hash API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify API key against its hash."""
    return hmac.compare_digest(hash_api_key(api_key), hashed_key)

# Session management
def generate_session_id() -> str:
    """Generate secure session ID."""
    return secrets.token_urlsafe(32)

def generate_csrf_token() -> str:
    """Generate CSRF token."""
    return secrets.token_urlsafe(32)

def verify_csrf_token(token: str, expected: str) -> bool:
    """Verify CSRF token."""
    return hmac.compare_digest(token, expected)

# Email validation
def validate_email_address(email: str) -> Dict[str, Union[bool, str]]:
    """Validate email address."""
    try:
        validation = validate_email(email)
        return {
            "is_valid": True,
            "normalized_email": validation.email
        }
    except EmailNotValidError as e:
        return {
            "is_valid": False,
            "error": str(e)
        }

# Rate limiting utilities
def generate_rate_limit_key(identifier: str, endpoint: str) -> str:
    """Generate rate limiting key."""
    return f"rate_limit:{identifier}:{endpoint}"

# Password reset token
def create_password_reset_token(email: str) -> str:
    """Create password reset token."""
    data = {
        "sub": email,
        "type": "password_reset",
        "exp": datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
    }
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify password reset token and return email."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") == "password_reset":
            return payload.get("sub")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        log_security_event("INVALID_RESET_TOKEN", details=str(e))
    return None

# Email verification token
def create_email_verification_token(email: str) -> str:
    """Create email verification token."""
    data = {
        "sub": email,
        "type": "email_verification",
        "exp": datetime.utcnow() + timedelta(days=7)  # 7 days expiry
    }
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def verify_email_verification_token(token: str) -> Optional[str]:
    """Verify email verification token and return email."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") == "email_verification":
            return payload.get("sub")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        log_security_event("INVALID_VERIFICATION_TOKEN", details=str(e))
    return None

# OAuth utilities
def generate_oauth_state() -> str:
    """Generate OAuth state parameter."""
    return secrets.token_urlsafe(32)

def verify_oauth_state(state: str, expected: str) -> bool:
    """Verify OAuth state parameter."""
    return hmac.compare_digest(state, expected)

# Security headers
def get_security_headers() -> Dict[str, str]:
    """Get security headers."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

# Input sanitization
def sanitize_input(input_string: str) -> str:
    """Sanitize user input."""
    if not input_string:
        return ""
    
    # Remove potential XSS characters
    dangerous_chars = ["<", ">", "&", "\"", "'", "/"]
    sanitized = input_string
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")
    
    return sanitized.strip()

# Permission utilities
class Permission:
    # User permissions
    READ_USERS = "read:users"
    WRITE_USERS = "write:users"
    DELETE_USERS = "delete:users"
    
    # Customer permissions
    READ_CUSTOMERS = "read:customers"
    WRITE_CUSTOMERS = "write:customers"
    DELETE_CUSTOMERS = "delete:customers"
    
    # Sales permissions
    READ_SALES = "read:sales"
    WRITE_SALES = "write:sales"
    DELETE_SALES = "delete:sales"
    
    # Analytics permissions
    READ_ANALYTICS = "read:analytics"
    WRITE_ANALYTICS = "write:analytics"
    
    # Admin permissions
    ADMIN_ACCESS = "admin:access"
    SYSTEM_CONFIG = "system:config"

def check_permissions(user_permissions: list, required_permission: str) -> bool:
    """Check if user has required permission."""
    return required_permission in user_permissions

# Security event types
class SecurityEvent:
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    LOGOUT = "LOGOUT"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    TOKEN_REFRESH = "TOKEN_REFRESH"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"

# Export security functions
__all__ = [
    "verify_password",
    "get_password_hash",
    "validate_password_strength",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "get_current_user_from_token",
    "verify_refresh_token",
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    "generate_session_id",
    "generate_csrf_token",
    "verify_csrf_token",
    "validate_email_address",
    "create_password_reset_token",
    "verify_password_reset_token",
    "create_email_verification_token",
    "verify_email_verification_token",
    "generate_oauth_state",
    "verify_oauth_state",
    "get_security_headers",
    "sanitize_input",
    "Permission",
    "check_permissions",
    "SecurityEvent"
]