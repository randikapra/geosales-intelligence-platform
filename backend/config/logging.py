# backend/config/logging.py
import logging
import logging.config
import os
from datetime import datetime
from typing import Dict, Any
from .settings import get_settings

settings = get_settings()

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# Custom formatter with colors for console output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

# JSON formatter for structured logging
class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""
    
    def format(self, record):
        import json
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
            
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

# Logging configuration dictionary
LOGGING_CONFIG: Dict[str, Any] = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'colored': {
            '()': ColoredFormatter,
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            '()': JSONFormatter
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG' if settings.DEBUG else 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'colored' if settings.DEBUG else 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'detailed',
            'filename': settings.LOG_FILE,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'detailed',
            'filename': 'logs/error.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        },
        'security_file': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'json' if not settings.DEBUG else 'detailed',
            'filename': 'logs/security.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
            'encoding': 'utf8'
        },
        'api_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'json' if not settings.DEBUG else 'detailed',
            'filename': 'logs/api.log',
            'maxBytes': 20971520,  # 20MB
            'backupCount': 7,
            'encoding': 'utf8'
        },
        'ml_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'detailed',
            'filename': 'logs/ml.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': settings.LOG_LEVEL,
            'propagate': False
        },
        'fastapi': {
            'handlers': ['console', 'api_file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn': {
            'handlers': ['console', 'api_file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn.access': {
            'handlers': ['api_file'],
            'level': 'INFO',
            'propagate': False
        },
        'sqlalchemy': {
            'handlers': ['file'],
            'level': 'WARNING',
            'propagate': False
        },
        'sqlalchemy.engine': {
            'handlers': ['file'],
            'level': 'INFO' if settings.DEBUG else 'WARNING',
            'propagate': False
        },
        'security': {
            'handlers': ['console', 'security_file'],
            'level': 'WARNING',
            'propagate': False
        },
        'ml_engine': {
            'handlers': ['console', 'ml_file'],
            'level': 'INFO',
            'propagate': False
        },
        'api': {
            'handlers': ['console', 'api_file'],
            'level': 'INFO',
            'propagate': False
        },
        'services': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'geosales': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG' if settings.DEBUG else 'INFO',
            'propagate': False
        }
    }
}

def setup_logging():
    """Setup logging configuration."""
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Set up specific loggers
    logging.getLogger("passlib").setLevel(logging.WARNING)
    logging.getLogger("bcrypt").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    if not settings.DEBUG:
        # Reduce noise in production
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)

def log_api_request(request_id: str, method: str, path: str, user_id: str = None):
    """Log API request information."""
    logger = get_logger("api")
    extra = {'request_id': request_id}
    if user_id:
        extra['user_id'] = user_id
    logger.info(f"{method} {path}", extra=extra)

def log_api_response(request_id: str, status_code: int, duration: float):
    """Log API response information."""
    logger = get_logger("api")
    extra = {'request_id': request_id, 'duration': duration}
    logger.info(f"Response: {status_code} ({duration:.3f}s)", extra=extra)

def log_security_event(event_type: str, user_id: str = None, details: str = None, ip_address: str = None):
    """Log security-related events."""
    logger = get_logger("security")
    extra = {}
    if user_id:
        extra['user_id'] = user_id
    if ip_address:
        extra['ip_address'] = ip_address
    
    message = f"Security Event: {event_type}"
    if details:
        message += f" - {details}"
    
    logger.warning(message, extra=extra)

def log_ml_operation(operation: str, model: str, duration: float = None, accuracy: float = None):
    """Log machine learning operations."""
    logger = get_logger("ml_engine")
    extra = {}
    if duration:
        extra['duration'] = duration
    if accuracy:
        extra['accuracy'] = accuracy
    
    message = f"ML Operation: {operation} - Model: {model}"
    if accuracy:
        message += f" - Accuracy: {accuracy:.3f}"
    
    logger.info(message, extra=extra)

def log_database_operation(operation: str, table: str, duration: float = None, row_count: int = None):
    """Log database operations."""
    logger = get_logger("database")
    extra = {}
    if duration:
        extra['duration'] = duration
    if row_count:
        extra['row_count'] = row_count
    
    message = f"DB Operation: {operation} - Table: {table}"
    if row_count:
        message += f" - Rows: {row_count}"
    
    logger.info(message, extra=extra)

# Performance logging decorator
def log_performance(logger_name: str = "performance"):
    """Decorator to log function performance."""
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {str(e)}")
                raise
        return wrapper
    return decorator

# Initialize logging
setup_logging()

# Export commonly used functions
__all__ = [
    "setup_logging",
    "get_logger", 
    "log_api_request",
    "log_api_response",
    "log_security_event",
    "log_ml_operation",
    "log_database_operation",
    "log_performance"
]