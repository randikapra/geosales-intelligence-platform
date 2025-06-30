from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.ext.declarative import as_declarative
from pydantic import BaseModel
from fastapi import HTTPException
import math

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        """
        CRUD object with default methods to Create, Read, Update, Delete (CRUD).
        """
        self.model = model

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """Get a single record by ID"""
        return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ) -> Dict[str, Any]:
        """
        Get multiple records with pagination, filtering, and sorting
        """
        query = db.query(self.model)
        
        # Apply filters
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field) and value is not None:
                    if isinstance(value, str) and "%" in value:
                        # Support partial matching with %
                        query = query.filter(getattr(self.model, field).like(value))
                    elif isinstance(value, list):
                        # Support IN operations for list values
                        query = query.filter(getattr(self.model, field).in_(value))
                    else:
                        query = query.filter(getattr(self.model, field) == value)
        
        # Apply sorting
        if sort_by and hasattr(self.model, sort_by):
            if sort_order.lower() == "desc":
                query = query.order_by(desc(getattr(self.model, sort_by)))
            else:
                query = query.order_by(asc(getattr(self.model, sort_by)))
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        items = query.offset(skip).limit(limit).all()
        
        # Calculate pagination metadata
        total_pages = math.ceil(total / limit) if limit > 0 else 1
        current_page = (skip // limit) + 1 if limit > 0 else 1
        
        return {
            "items": items,
            "total": total,
            "page": current_page,
            "pages": total_pages,
            "per_page": limit,
            "has_next": current_page < total_pages,
            "has_prev": current_page > 1
        }

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """Create a new record"""
        obj_in_data = obj_in.dict()
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def create_bulk(self, db: Session, *, objs_in: List[CreateSchemaType]) -> List[ModelType]:
        """Create multiple records in bulk"""
        db_objs = []
        for obj_in in objs_in:
            obj_in_data = obj_in.dict()
            db_obj = self.model(**obj_in_data)
            db_objs.append(db_obj)
        
        db.add_all(db_objs)
        db.commit()
        for db_obj in db_objs:
            db.refresh(db_obj)
        return db_objs

    def update(
        self,
        db: Session,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """Update an existing record"""
        obj_data = obj_in.dict(exclude_unset=True) if isinstance(obj_in, BaseModel) else obj_in
        
        for field, value in obj_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, *, id: int) -> ModelType:
        """Delete a record by ID"""
        obj = db.query(self.model).get(id)
        if not obj:
            raise HTTPException(status_code=404, detail="Record not found")
        db.delete(obj)
        db.commit()
        return obj

    def delete_multi(self, db: Session, *, ids: List[int]) -> int:
        """Delete multiple records by IDs"""
        count = db.query(self.model).filter(self.model.id.in_(ids)).delete(synchronize_session=False)
        db.commit()
        return count

    def search(
        self,
        db: Session,
        *,
        search_term: str,
        search_fields: List[str],
        skip: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Full-text search across specified fields
        """
        query = db.query(self.model)
        
        if search_term and search_fields:
            search_conditions = []
            for field in search_fields:
                if hasattr(self.model, field):
                    search_conditions.append(
                        getattr(self.model, field).ilike(f"%{search_term}%")
                    )
            
            if search_conditions:
                query = query.filter(or_(*search_conditions))
        
        total = query.count()
        items = query.offset(skip).limit(limit).all()
        
        total_pages = math.ceil(total / limit) if limit > 0 else 1
        current_page = (skip // limit) + 1 if limit > 0 else 1
        
        return {
            "items": items,
            "total": total,
            "page": current_page,
            "pages": total_pages,
            "per_page": limit,
            "has_next": current_page < total_pages,
            "has_prev": current_page > 1
        }

    def get_by_field(self, db: Session, field: str, value: Any) -> Optional[ModelType]:
        """Get a record by any field"""
        if not hasattr(self.model, field):
            return None
        return db.query(self.model).filter(getattr(self.model, field) == value).first()

    def get_multi_by_field(
        self,
        db: Session,
        field: str,
        value: Any,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Get multiple records by field value"""
        if not hasattr(self.model, field):
            return []
        return (
            db.query(self.model)
            .filter(getattr(self.model, field) == value)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def count(self, db: Session, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filters"""
        query = db.query(self.model)
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field) and value is not None:
                    query = query.filter(getattr(self.model, field) == value)
        
        return query.count()

    def exists(self, db: Session, id: Any) -> bool:
        """Check if record exists by ID"""
        return db.query(self.model).filter(self.model.id == id).first() is not None

    def get_or_create(
        self,
        db: Session,
        *,
        defaults: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tuple[ModelType, bool]:
        """Get existing record or create new one"""
        obj = db.query(self.model).filter_by(**kwargs).first()
        if obj:
            return obj, False
        
        create_data = {**kwargs, **(defaults or {})}
        obj = self.model(**create_data)
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj, True

    def aggregate(
        self,
        db: Session,
        *,
        group_by: str,
        aggregations: Dict[str, str],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform aggregations grouped by a field
        aggregations: {"field_name": "function"} where function can be sum, avg, count, min, max
        """
        if not hasattr(self.model, group_by):
            return []
        
        # Start with group by field
        select_fields = [getattr(self.model, group_by).label(group_by)]
        
        # Add aggregation fields
        for field, func in aggregations.items():
            if hasattr(self.model, field):
                if func.lower() == "sum":
                    select_fields.append(func.sum(getattr(self.model, field)).label(f"{field}_{func}"))
                elif func.lower() == "avg":
                    select_fields.append(func.avg(getattr(self.model, field)).label(f"{field}_{func}"))
                elif func.lower() == "count":
                    select_fields.append(func.count(getattr(self.model, field)).label(f"{field}_{func}"))
                elif func.lower() == "min":
                    select_fields.append(func.min(getattr(self.model, field)).label(f"{field}_{func}"))
                elif func.lower() == "max":
                    select_fields.append(func.max(getattr(self.model, field)).label(f"{field}_{func}"))
        
        query = db.query(*select_fields)
        
        # Apply filters
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field) and value is not None:
                    query = query.filter(getattr(self.model, field) == value)
        
        # Group by
        query = query.group_by(getattr(self.model, group_by))
        
        # Execute and format results
        results = []
        for row in query.all():
            result_dict = {}
            for i, field in enumerate([group_by] + [f"{f}_{func}" for f, func in aggregations.items()]):
                result_dict[field] = row[i] if i < len(row) else None
            results.append(result_dict)
        
        return results


class BaseRepository:
    """Base repository class with common database operations"""
    
    def __init__(self, db: Session):
        self.db = db

    def execute_raw_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL query and return results as dictionaries"""
        result = self.db.execute(text(query), params or {})
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about table structure"""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = :table_name
        ORDER BY ordinal_position;
        """
        return self.execute_raw_query(query, {"table_name": table_name})