"""
Custom exceptions for SFA (Sales Force Automation) system.
Handles HTTP exceptions, validation errors, and business logic errors.
"""

from typing import Any, Dict, Optional, List
from fastapi import HTTPException, status
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Standard error detail structure"""
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class BaseCustomException(HTTPException):
    """Base class for all custom exceptions"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        field: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.field = field


# =============================================================================
# HTTP EXCEPTIONS
# =============================================================================

class NotFoundError(BaseCustomException):
    """Resource not found exception"""
    
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with identifier '{identifier}' not found",
            error_code="RESOURCE_NOT_FOUND"
        )


class UnauthorizedError(BaseCustomException):
    """Unauthorized access exception"""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message,
            error_code="UNAUTHORIZED",
            headers={"WWW-Authenticate": "Bearer"}
        )


class ForbiddenError(BaseCustomException):
    """Forbidden access exception"""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message,
            error_code="FORBIDDEN"
        )


class ConflictError(BaseCustomException):
    """Resource conflict exception"""
    
    def __init__(self, message: str, resource: str = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=message,
            error_code="RESOURCE_CONFLICT"
        )


class BadRequestError(BaseCustomException):
    """Bad request exception"""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
            error_code="BAD_REQUEST",
            field=field
        )


class InternalServerError(BaseCustomException):
    """Internal server error exception"""
    
    def __init__(self, message: str = "Internal server error occurred"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
            error_code="INTERNAL_SERVER_ERROR"
        )


class ServiceUnavailableError(BaseCustomException):
    """Service unavailable exception"""
    
    def __init__(self, service: str = "Service"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{service} is temporarily unavailable",
            error_code="SERVICE_UNAVAILABLE"
        )


# =============================================================================
# VALIDATION ERRORS
# =============================================================================

class ValidationError(BaseCustomException):
    """Base validation error"""
    
    def __init__(self, message: str, field: str = None, errors: List[ErrorDetail] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
            error_code="VALIDATION_ERROR",
            field=field
        )
        self.errors = errors or []


class InvalidCoordinatesError(ValidationError):
    """Invalid GPS coordinates error"""
    
    def __init__(self, latitude: float = None, longitude: float = None):
        if latitude is not None and longitude is not None:
            message = f"Invalid coordinates: latitude={latitude}, longitude={longitude}"
        else:
            message = "Invalid GPS coordinates provided"
        
        super().__init__(
            message=message,
            field="coordinates",
            errors=[
                ErrorDetail(
                    code="INVALID_COORDINATES",
                    message="Latitude must be between -90 and 90, longitude between -180 and 180",
                    field="coordinates"
                )
            ]
        )


class InvalidDateRangeError(ValidationError):
    """Invalid date range error"""
    
    def __init__(self, start_date: str = None, end_date: str = None):
        message = "Invalid date range: start date must be before end date"
        if start_date and end_date:
            message = f"Invalid date range: {start_date} to {end_date}"
        
        super().__init__(
            message=message,
            field="date_range",
            errors=[
                ErrorDetail(
                    code="INVALID_DATE_RANGE",
                    message=message,
                    field="date_range"
                )
            ]
        )


class InvalidOrderValueError(ValidationError):
    """Invalid order value error"""
    
    def __init__(self, value: float):
        super().__init__(
            message=f"Invalid order value: {value}. Must be greater than 0",
            field="order_value",
            errors=[
                ErrorDetail(
                    code="INVALID_ORDER_VALUE",
                    message="Order value must be greater than 0",
                    field="order_value",
                    details={"provided_value": value}
                )
            ]
        )


class InvalidUserCodeError(ValidationError):
    """Invalid user code format error"""
    
    def __init__(self, user_code: str):
        super().__init__(
            message=f"Invalid user code format: {user_code}",
            field="user_code",
            errors=[
                ErrorDetail(
                    code="INVALID_USER_CODE",
                    message="User code must follow the pattern: [A-Z]{2,3}-[0-9]{6}",
                    field="user_code",
                    details={"provided_code": user_code}
                )
            ]
        )


class InvalidDistributorCodeError(ValidationError):
    """Invalid distributor code format error"""
    
    def __init__(self, distributor_code: str):
        super().__init__(
            message=f"Invalid distributor code format: {distributor_code}",
            field="distributor_code",
            errors=[
                ErrorDetail(
                    code="INVALID_DISTRIBUTOR_CODE",
                    message="Distributor code format is invalid",
                    field="distributor_code",
                    details={"provided_code": distributor_code}
                )
            ]
        )


# =============================================================================
# BUSINESS LOGIC ERRORS
# =============================================================================

class BusinessLogicError(BaseCustomException):
    """Base business logic error"""
    
    def __init__(self, message: str, error_code: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
            error_code=error_code
        )


class OrderSubmissionError(BusinessLogicError):
    """Order submission business logic error"""
    
    def __init__(self, reason: str, order_code: str = None):
        message = f"Order submission failed: {reason}"
        if order_code:
            message = f"Order {order_code} submission failed: {reason}"
        
        super().__init__(
            message=message,
            error_code="ORDER_SUBMISSION_FAILED"
        )


class DuplicateOrderError(BusinessLogicError):
    """Duplicate order error"""
    
    def __init__(self, order_code: str):
        super().__init__(
            message=f"Order {order_code} already exists",
            error_code="DUPLICATE_ORDER"
        )


class InvalidTourCodeError(BusinessLogicError):
    """Invalid tour code for user"""
    
    def __init__(self, user_code: str, tour_code: str):
        super().__init__(
            message=f"Tour code {tour_code} is not valid for user {user_code}",
            error_code="INVALID_TOUR_CODE"
        )


class UserNotInTerritoryError(BusinessLogicError):
    """User not authorized for territory"""
    
    def __init__(self, user_code: str, territory_code: str):
        super().__init__(
            message=f"User {user_code} is not authorized for territory {territory_code}",
            error_code="USER_NOT_IN_TERRITORY"
        )


class GPSTrackingError(BusinessLogicError):
    """GPS tracking related errors"""
    
    def __init__(self, user_code: str, reason: str):
        super().__init__(
            message=f"GPS tracking error for user {user_code}: {reason}",
            error_code="GPS_TRACKING_ERROR"
        )


class OutOfGeofenceError(BusinessLogicError):
    """User is outside allowed geofence"""
    
    def __init__(self, user_code: str, latitude: float, longitude: float):
        super().__init__(
            message=f"User {user_code} is outside allowed geofence at coordinates ({latitude}, {longitude})",
            error_code="OUT_OF_GEOFENCE"
        )


class InactiveCustomerError(BusinessLogicError):
    """Customer is inactive"""
    
    def __init__(self, customer_code: str):
        super().__init__(
            message=f"Customer {customer_code} is inactive and cannot place orders",
            error_code="INACTIVE_CUSTOMER"
        )


class InsufficientStockError(BusinessLogicError):
    """Insufficient stock for order"""
    
    def __init__(self, product_code: str, requested_qty: int, available_qty: int):
        super().__init__(
            message=f"Insufficient stock for product {product_code}. Requested: {requested_qty}, Available: {available_qty}",
            error_code="INSUFFICIENT_STOCK"
        )


class OrderValueExceedsLimitError(BusinessLogicError):
    """Order value exceeds customer credit limit"""
    
    def __init__(self, customer_code: str, order_value: float, credit_limit: float):
        super().__init__(
            message=f"Order value {order_value} exceeds credit limit {credit_limit} for customer {customer_code}",
            error_code="ORDER_VALUE_EXCEEDS_LIMIT"
        )


class InvalidOrderStatusError(BusinessLogicError):
    """Invalid order status transition"""
    
    def __init__(self, order_code: str, current_status: str, requested_status: str):
        super().__init__(
            message=f"Cannot change order {order_code} status from {current_status} to {requested_status}",
            error_code="INVALID_ORDER_STATUS_TRANSITION"
        )


class DatabaseConnectionError(InternalServerError):
    """Database connection error"""
    
    def __init__(self, operation: str = "database operation"):
        super().__init__(
            message=f"Database connection failed during {operation}"
        )
        self.error_code = "DATABASE_CONNECTION_ERROR"


class ExternalServiceError(ServiceUnavailableError):
    """External service error"""
    
    def __init__(self, service_name: str, operation: str = None):
        message = f"External service {service_name} is unavailable"
        if operation:
            message = f"External service {service_name} failed during {operation}"
        
        super().__init__(service=service_name)
        self.error_code = "EXTERNAL_SERVICE_ERROR"


class RateLimitExceededError(BaseCustomException):
    """Rate limit exceeded error"""
    
    def __init__(self, limit: int, window: str = "minute"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per {window}",
            error_code="RATE_LIMIT_EXCEEDED",
            headers={"Retry-After": "60"}
        )


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

def format_error_response(error: BaseCustomException) -> Dict[str, Any]:
    """Format error response for consistent API responses"""
    response = {
        "error": True,
        "error_code": getattr(error, 'error_code', 'UNKNOWN_ERROR'),
        "message": error.detail,
        "status_code": error.status_code
    }
    
    if hasattr(error, 'field') and error.field:
        response["field"] = error.field
    
    if hasattr(error, 'errors') and error.errors:
        response["errors"] = [
            {
                "code": err.code,
                "message": err.message,
                "field": err.field,
                "details": err.details
            } for err in error.errors
        ]
    
    return response


def get_exception_by_code(error_code: str) -> type:
    """Get exception class by error code"""
    exception_mapping = {
        "RESOURCE_NOT_FOUND": NotFoundError,
        "UNAUTHORIZED": UnauthorizedError,
        "FORBIDDEN": ForbiddenError,
        "RESOURCE_CONFLICT": ConflictError,
        "BAD_REQUEST": BadRequestError,
        "VALIDATION_ERROR": ValidationError,
        "INVALID_COORDINATES": InvalidCoordinatesError,
        "INVALID_DATE_RANGE": InvalidDateRangeError,
        "INVALID_ORDER_VALUE": InvalidOrderValueError,
        "INVALID_USER_CODE": InvalidUserCodeError,
        "INVALID_DISTRIBUTOR_CODE": InvalidDistributorCodeError,
        "ORDER_SUBMISSION_FAILED": OrderSubmissionError,
        "DUPLICATE_ORDER": DuplicateOrderError,
        "INVALID_TOUR_CODE": InvalidTourCodeError,
        "USER_NOT_IN_TERRITORY": UserNotInTerritoryError,
        "GPS_TRACKING_ERROR": GPSTrackingError,
        "OUT_OF_GEOFENCE": OutOfGeofenceError,
        "INACTIVE_CUSTOMER": InactiveCustomerError,
        "INSUFFICIENT_STOCK": InsufficientStockError,
        "ORDER_VALUE_EXCEEDS_LIMIT": OrderValueExceedsLimitError,
        "INVALID_ORDER_STATUS_TRANSITION": InvalidOrderStatusError,
        "DATABASE_CONNECTION_ERROR": DatabaseConnectionError,
        "EXTERNAL_SERVICE_ERROR": ExternalServiceError,
        "RATE_LIMIT_EXCEEDED": RateLimitExceededError,
        "INTERNAL_SERVER_ERROR": InternalServerError,
        "SERVICE_UNAVAILABLE": ServiceUnavailableError
    }
    
    return exception_mapping.get(error_code, BaseCustomException)