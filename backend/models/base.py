# # # backend/models/base.py
# # """
# # Base model with common fields
# # """
# # from sqlalchemy import Column, Integer, DateTime, Boolean
# # from sqlalchemy.sql import func
# # from sqlalchemy.ext.declarative import declarative_base

# # Base = declarative_base()


# # class BaseModel(Base):
# #     __abstract__ = True
    
# #     id = Column(Integer, primary_key=True, index=True)
# #     created_at = Column(DateTime(timezone=True), server_default=func.now())
# #     updated_at = Column(DateTime(timezone=True), onupdate=func.now())
# #     is_active = Column(Boolean, default=True)

# """
# Base SQLAlchemy model with common fields and functionality.
# """
# from datetime import datetime
# from typing import Any, Dict
# from sqlalchemy import Column, Integer, DateTime, String, Boolean
# from sqlalchemy.ext.declarative import declarative_base, declared_attr
# from sqlalchemy.orm import Session
# from sqlalchemy.sql import func

# Base = declarative_base()


# class BaseModel(Base):
#     """Abstract base model with common fields and methods."""
    
#     __abstract__ = True
    
#     id = Column(Integer, primary_key=True, index=True, autoincrement=True)
#     created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
#     updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
#     is_active = Column(Boolean, default=True, nullable=False)
#     created_by = Column(String(100), nullable=True)
#     updated_by = Column(String(100), nullable=True)
    
#     @declared_attr
#     def __tablename__(cls):
#         """Generate table name from class name."""
#         return cls.__name__.lower()
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert model instance to dictionary."""
#         return {
#             column.name: getattr(self, column.name)
#             for column in self.__table__.columns
#         }
    
#     def update_fields(self, **kwargs) -> None:
#         """Update model fields from keyword arguments."""
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#         self.updated_at = datetime.utcnow()
    
#     @classmethod
#     def create(cls, db: Session, **kwargs):
#         """Create new instance and save to database."""
#         instance = cls(**kwargs)
#         db.add(instance)
#         db.commit()
#         db.refresh(instance)
#         return instance
    
#     def save(self, db: Session):
#         """Save current instance to database."""
#         db.add(self)
#         db.commit()
#         db.refresh(self)
#         return self
    
#     def delete(self, db: Session, soft_delete: bool = True):
#         """Delete instance (soft delete by default)."""
#         if soft_delete:
#             self.is_active = False
#             self.updated_at = datetime.utcnow()
#             db.commit()
#         else:
#             db.delete(self)
#             db.commit()
    
#     def __repr__(self):
#         return f"<{self.__class__.__name__}(id={self.id})>"


# class TimestampMixin:
#     """Mixin for models that need timestamp tracking."""
    
#     created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
#     updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


# class SoftDeleteMixin:
#     """Mixin for models that support soft delete."""
    
#     is_active = Column(Boolean, default=True, nullable=False)
#     deleted_at = Column(DateTime(timezone=True), nullable=True)
    
#     def soft_delete(self, db: Session):
#         """Perform soft delete."""
#         self.is_active = False
#         self.deleted_at = datetime.utcnow()
#         db.commit()


# class AuditMixin:
#     """Mixin for models that need audit trail."""
    
#     created_by = Column(String(100), nullable=True)
#     updated_by = Column(String(100), nullable=True)
#     version = Column(Integer, default=1, nullable=False)
    
#     def increment_version(self):
#         """Increment version for optimistic locking."""
#         self.version += 1



"""
Base SQLAlchemy model with common fields and utilities.
"""
from datetime import datetime
from typing import Any, Dict
from sqlalchemy import Column, Integer, DateTime, String, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class BaseModel(Base):
    """
    Abstract base model with common fields and methods for all models.
    """
    __abstract__ = True
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # UUID for external references
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Soft delete functionality
    is_active = Column(Boolean, default=True, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Audit fields
    created_by = Column(String(100), nullable=True)
    updated_by = Column(String(100), nullable=True)
    
    # Metadata for additional information
    metadata_json = Column(Text, nullable=True)  # JSON string for flexible metadata
    
    # Version control for optimistic locking
    version = Column(Integer, default=1, nullable=False)
    
    @hybrid_property
    def is_valid(self):
        """Check if record is active and not deleted."""
        return self.is_active and not self.is_deleted
    
    def soft_delete(self, deleted_by: str = None):
        """Soft delete the record."""
        self.is_deleted = True
        self.is_active = False
        self.deleted_at = datetime.utcnow()
        self.updated_by = deleted_by
        self.version += 1
    
    def restore(self, restored_by: str = None):
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.is_active = True
        self.deleted_at = None
        self.updated_by = restored_by
        self.version += 1
    
    def update_metadata(self, key: str, value: Any):
        """Update metadata with key-value pair."""
        import json
        if self.metadata_json:
            metadata = json.loads(self.metadata_json)
        else:
            metadata = {}
        metadata[key] = value
        self.metadata_json = json.dumps(metadata)
    
    def get_metadata(self, key: str = None):
        """Get metadata value by key or all metadata."""
        import json
        if not self.metadata_json:
            return None if key else {}
        metadata = json.loads(self.metadata_json)
        return metadata.get(key) if key else metadata
    
    def to_dict(self, include_deleted: bool = False) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        
        # Exclude deleted records unless explicitly requested
        if not include_deleted and self.is_deleted:
            return {}
            
        return result
    
    def update_from_dict(self, data: Dict[str, Any], updated_by: str = None):
        """Update model instance from dictionary."""
        for key, value in data.items():
            if hasattr(self, key) and key not in ['id', 'uuid', 'created_at', 'version']:
                setattr(self, key, value)
        
        self.updated_by = updated_by
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    @classmethod
    def get_active_query(cls, session):
        """Get query for active (non-deleted) records."""
        return session.query(cls).filter(
            cls.is_active == True,
            cls.is_deleted == False
        )
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, uuid={self.uuid})>"


class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    is_active = Column(Boolean, default=True, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)


class AuditMixin:
    """Mixin for audit fields."""
    created_by = Column(String(100), nullable=True)
    updated_by = Column(String(100), nullable=True)


class MetadataMixin:
    """Mixin for metadata functionality."""
    metadata_json = Column(Text, nullable=True)
    
    def update_metadata(self, key: str, value: Any):
        """Update metadata with key-value pair."""
        import json
        if self.metadata_json:
            metadata = json.loads(self.metadata_json)
        else:
            metadata = {}
        metadata[key] = value
        self.metadata_json = json.dumps(metadata)
    
    def get_metadata(self, key: str = None):
        """Get metadata value by key or all metadata."""
        import json
        if not self.metadata_json:
            return None if key else {}
        metadata = json.loads(self.metadata_json)
        return metadata.get(key) if key else metadata