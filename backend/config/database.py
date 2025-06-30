# backend/config/database.py
from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import Engine
import sqlite3
import logging
from typing import Generator
from .settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Database engine configuration
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,
    } if "sqlite" in settings.DATABASE_URL else {
        "connect_timeout": 30,
        "server_settings": {
            "jit": "off"
        }
    }
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()

# Metadata for database introspection
metadata = MetaData()

# Database session dependency
def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency.
    Creates a new database session for each request.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Connection pooling configuration
def configure_connection_pool():
    """Configure connection pool settings based on environment."""
    if settings.DEBUG:
        # Development settings
        engine.pool._recycle = 3600  # 1 hour
        engine.pool._pool_size = 5
        engine.pool._max_overflow = 10
    else:
        # Production settings
        engine.pool._recycle = 300   # 5 minutes
        engine.pool._pool_size = 20
        engine.pool._max_overflow = 50

# SQLite-specific configuration
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=memory")
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
        cursor.close()

# PostgreSQL-specific configuration
@event.listens_for(Engine, "connect")
def set_postgresql_search_path(dbapi_connection, connection_record):
    """Set search path for PostgreSQL."""
    if hasattr(dbapi_connection, 'set_session'):
        dbapi_connection.set_session(autocommit=True)

# Database health check
def check_database_health() -> bool:
    """Check if database connection is healthy."""
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

# Database initialization
def init_database():
    """Initialize database with tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# Database cleanup
def cleanup_database():
    """Clean up database connections."""
    try:
        engine.dispose()
        logger.info("Database connections cleaned up")
    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")

# Connection testing
def test_connection():
    """Test database connection."""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

# Transaction context manager
class DatabaseTransaction:
    """Context manager for database transactions."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def __enter__(self):
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.db.rollback()
        else:
            self.db.commit()

# Async database support (for future use)
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker as async_sessionmaker
    
    async_engine = create_async_engine(
        settings.DATABASE_URL_ASYNC,
        echo=settings.DEBUG,
        pool_pre_ping=True
    )
    
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async def get_async_db() -> Generator[AsyncSession, None, None]:
        """Async database session dependency."""
        async with AsyncSessionLocal() as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Async database session error: {e}")
                await session.rollback()
                raise
            finally:
                await session.close()
                
except ImportError:
    logger.warning("Async database support not available")
    async_engine = None
    AsyncSessionLocal = None

# Database URL utilities
def get_database_url(env: str = "development") -> str:
    """Get database URL for specific environment."""
    if env == "testing":
        return "sqlite:///./test.db"
    elif env == "production":
        return settings.DATABASE_URL
    else:
        return settings.DATABASE_URL

# Initialize connection pool
configure_connection_pool()

# Export commonly used objects
__all__ = [
    "engine",
    "SessionLocal",
    "Base",
    "get_db",
    "init_database",
    "cleanup_database",
    "check_database_health",
    "DatabaseTransaction",
    "async_engine",
    "AsyncSessionLocal",
    "get_async_db"
]