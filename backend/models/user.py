# """
# User model for authentication, roles, and permissions.
# """
# import enum
# from datetime import datetime
# from typing import List, Optional
# from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum, Text, ForeignKey, Table
# from sqlalchemy.orm import relationship
# from sqlalchemy.ext.hybrid import hybrid_property
# from passlib.context import CryptContext
# from .base import BaseModel, Base

# # Password hashing context
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # Association table for user roles (many-to-many)
# user_roles = Table(
#     'user_roles',
#     Base.metadata,
#     Column('user_id', Integer, ForeignKey('user.id'), primary_key=True),
#     Column('role_id', Integer, ForeignKey('role.id'), primary_key=True)
# )

# # Association table for role permissions (many-to-many)
# role_permissions = Table(
#     'role_permissions',
#     Base.metadata,
#     Column('role_id', Integer, ForeignKey('role.id'), primary_key=True),
#     Column('permission_id', Integer, ForeignKey('permission.id'), primary_key=True)
# )


# class UserStatus(enum.Enum):
#     """User account status enumeration."""
#     ACTIVE = "active"
#     INACTIVE = "inactive"
#     SUSPENDED = "suspended"
#     PENDING = "pending"


# class UserType(enum.Enum):
#     """User type enumeration."""
#     ADMIN = "admin"
#     MANAGER = "manager"
#     SALES_REP = "sales_rep"
#     DEALER = "dealer"
#     ANALYST = "analyst"
#     VIEWER = "viewer"


# class User(BaseModel):
#     """User model for authentication and authorization."""
    
#     __tablename__ = "user"
    
#     # Basic Information
#     username = Column(String(50), unique=True, index=True, nullable=False)
#     email = Column(String(100), unique=True, index=True, nullable=False)
#     hashed_password = Column(String(255), nullable=False)
#     full_name = Column(String(100), nullable=False)
#     phone = Column(String(20), nullable=True)
    
#     # User Status and Type
#     status = Column(Enum(UserStatus), default=UserStatus.ACTIVE, nullable=False)
#     user_type = Column(Enum(UserType), default=UserType.VIEWER, nullable=False)
    
#     # Authentication
#     is_superuser = Column(Boolean, default=False, nullable=False)
#     is_verified = Column(Boolean, default=False, nullable=False)
#     email_verified_at = Column(DateTime(timezone=True), nullable=True)
#     last_login = Column(DateTime(timezone=True), nullable=True)
#     login_attempts = Column(Integer, default=0, nullable=False)
#     locked_until = Column(DateTime(timezone=True), nullable=True)
    
#     # Profile Information
#     avatar_url = Column(String(255), nullable=True)
#     bio = Column(Text, nullable=True)
#     timezone = Column(String(50), default="UTC", nullable=False)
#     language = Column(String(10), default="en", nullable=False)
    
#     # Security
#     password_changed_at = Column(DateTime(timezone=True), nullable=True)
#     reset_token = Column(String(255), nullable=True)
#     reset_token_expires = Column(DateTime(timezone=True), nullable=True)
#     verification_token = Column(String(255), nullable=True)
    
#     # Business Context
#     employee_id = Column(String(50), unique=True, nullable=True)
#     department = Column(String(100), nullable=True)
#     territory_code = Column(String(20), nullable=True)
#     division_code = Column(String(20), nullable=True)
    
#     # Relationships
#     roles = relationship("Role", secondary=user_roles, back_populates="users")
#     sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
#     created_sales = relationship("Sales", foreign_keys="Sales.created_by_id", back_populates="created_by_user")
    
#     @hybrid_property
#     def is_active_user(self):
#         """Check if user is active and not locked."""
#         if not self.is_active or self.status != UserStatus.ACTIVE:
#             return False
#         if self.locked_until and self.locked_until > datetime.utcnow():
#             return False
#         return True
    
#     def set_password(self, password: str) -> None:
#         """Hash and set password."""
#         self.hashed_password = pwd_context.hash(password)
#         self.password_changed_at = datetime.utcnow()
    
#     def verify_password(self, password: str) -> bool:
#         """Verify password against hash."""
#         return pwd_context.verify(password, self.hashed_password)
    
#     def has_permission(self, permission_name: str) -> bool:
#         """Check if user has specific permission."""
#         if self.is_superuser:
#             return True
        
#         for role in self.roles:
#             if role.has_permission(permission_name):
#                 return True
#         return False
    
#     def has_role(self, role_name: str) -> bool:
#         """Check if user has specific role."""
#         return any(role.name == role_name for role in self.roles)
    
#     def lock_account(self, minutes: int = 30) -> None:
#         """Lock user account for specified minutes."""
#         self.locked_until = datetime.utcnow() + datetime.timedelta(minutes=minutes)
#         self.login_attempts = 0
    
#     def unlock_account(self) -> None:
#         """Unlock user account."""
#         self.locked_until = None
#         self.login_attempts = 0
    
#     def increment_login_attempts(self) -> None:
#         """Increment failed login attempts."""
#         self.login_attempts += 1
#         if self.login_attempts >= 5:  # Lock after 5 failed attempts
#             self.lock_account()
    
#     def reset_login_attempts(self) -> None:
#         """Reset login attempts counter."""
#         self.login_attempts = 0
#         self.last_login = datetime.utcnow()


# class Role(BaseModel):
#     """Role model for user authorization."""
    
#     __tablename__ = "role"
    
#     name = Column(String(50), unique=True, nullable=False, index=True)
#     display_name = Column(String(100), nullable=False)
#     description = Column(Text, nullable=True)
#     is_system_role = Column(Boolean, default=False, nullable=False)
    
#     # Relationships
#     users = relationship("User", secondary=user_roles, back_populates="roles")
#     permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    
#     def has_permission(self, permission_name: str) -> bool:
#         """Check if role has specific permission."""
#         return any(perm.name == permission_name for perm in self.permissions)
    
#     def add_permission(self, permission: 'Permission') -> None:
#         """Add permission to role."""
#         if permission not in self.permissions:
#             self.permissions.append(permission)
    
#     def remove_permission(self, permission: 'Permission') -> None:
#         """Remove permission from role."""
#         if permission in self.permissions:
#             self.permissions.remove(permission)


# class Permission(BaseModel):
#     """Permission model for fine-grained access control."""
    
#     __tablename__ = "permission"
    
#     name = Column(String(100), unique=True, nullable=False, index=True)
#     display_name = Column(String(150), nullable=False)
#     description = Column(Text, nullable=True)
#     resource = Column(String(50), nullable=False)  # e.g., 'customer', 'sales', 'reports'
#     action = Column(String(20), nullable=False)    # e.g., 'create', 'read', 'update', 'delete'
    
#     # Relationships
#     roles = relationship("Role", secondary=role_permissions, back_populates="permissions")
    
#     @classmethod
#     def create_crud_permissions(cls, resource: str) -> List['Permission']:
#         """Create standard CRUD permissions for a resource."""
#         actions = ['create', 'read', 'update', 'delete']
#         permissions = []
        
#         for action in actions:
#             perm = cls(
#                 name=f"{resource}:{action}",
#                 display_name=f"{action.title()} {resource.title()}",
#                 description=f"Permission to {action} {resource} records",
#                 resource=resource,
#                 action=action
#             )
#             permissions.append(perm)
        
#         return permissions


# class UserSession(BaseModel):
#     """User session tracking model."""
    
#     __tablename__ = "user_session"
    
#     user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
#     session_token = Column(String(255), unique=True, nullable=False, index=True)
#     refresh_token = Column(String(255), unique=True, nullable=True, index=True)
#     expires_at = Column(DateTime(timezone=True), nullable=False)
#     refresh_expires_at = Column(DateTime(timezone=True), nullable=True)
    
#     # Session metadata
#     ip_address = Column(String(45), nullable=True)  # IPv6 compatible
#     user_agent = Column(Text, nullable=True)
#     device_info = Column(Text, nullable=True)
#     location = Column(String(100), nullable=True)
    
#     # Session status
#     is_revoked = Column(Boolean, default=False, nullable=False)
#     revoked_at = Column(DateTime(timezone=True), nullable=True)
#     last_activity = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
#     # Relationships
#     user = relationship("User", back_populates="sessions")
    
#     @property
#     def is_valid(self) -> bool:
#         """Check if session is still valid."""
#         if self.is_revoked:
#             return False
#         if self.expires_at < datetime.utcnow():
#             return False
#         return True
    
#     def revoke(self) -> None:
#         """Revoke the session."""
#         self.is_revoked = True
#         self.revoked_at = datetime.utcnow()
    
#     def refresh_activity(self) -> None:
#         """Update last activity timestamp."""
#         self.last_activity = datetime.utcnow()


# class UserLoginHistory(BaseModel):
#     """User login history tracking."""
    
#     __tablename__ = "user_login_history"
    
#     user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
#     login_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
#     logout_time = Column(DateTime(timezone=True), nullable=True)
#     ip_address = Column(String(45), nullable=True)
#     user_agent = Column(Text, nullable=True)
#     success = Column(Boolean, nullable=False)
#     failure_reason = Column(String(100), nullable=True)
#     session_duration = Column(Integer, nullable=True)  # in seconds
    
#     # Relationships
#     user = relationship("User")
    
#     def calculate_session_duration(self) -> Optional[int]:
#         """Calculate session duration in seconds."""
#         if self.logout_time:
#             delta = self.logout_time - self.login_time
#             return int(delta.total_seconds())
#         return None

"""
User model for authentication, roles, and permissions.
"""
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Table, Enum
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
import enum
import hashlib
import secrets
from werkzeug.security import generate_password_hash, check_password_hash

from .base import BaseModel, Base


class UserRole(enum.Enum):
    """User roles enumeration."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    SALES_MANAGER = "sales_manager"
    TERRITORY_MANAGER = "territory_manager"
    DEALER = "dealer"
    SALES_REP = "sales_rep"
    ANALYST = "analyst"
    VIEWER = "viewer"


class UserStatus(enum.Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    LOCKED = "locked"


# Association table for user-permission many-to-many relationship
user_permissions = Table(
    'user_permissions',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('permission_id', Integer, ForeignKey('permissions.id'), primary_key=True),
    Column('granted_at', DateTime(timezone=True), server_default=func.now()),
    Column('granted_by', Integer, ForeignKey('users.id'))
)

# Association table for user-role many-to-many relationship
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True),
    Column('assigned_at', DateTime(timezone=True), server_default=func.now()),
    Column('assigned_by', Integer, ForeignKey('users.id'))
)


class Permission(BaseModel):
    """Permission model for fine-grained access control."""
    __tablename__ = 'permissions'
    
    name = Column(String(100), unique=True, nullable=False, index=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(500), nullable=True)
    resource = Column(String(100), nullable=False)  # e.g., 'customers', 'sales', 'reports'
    action = Column(String(50), nullable=False)  # e.g., 'create', 'read', 'update', 'delete'
    
    def __repr__(self):
        return f"<Permission(code={self.code}, resource={self.resource}, action={self.action})>"


class Role(BaseModel):
    """Role model for grouping permissions."""
    __tablename__ = 'roles'
    
    name = Column(String(100), unique=True, nullable=False, index=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(500), nullable=True)
    is_system_role = Column(Boolean, default=False, nullable=False)  # System roles cannot be deleted
    
    # Many-to-many relationship with permissions
    permissions = relationship(
        "Permission",
        secondary="role_permissions",
        back_populates="roles",
        lazy="dynamic"
    )
    
    def has_permission(self, permission_code: str) -> bool:
        """Check if role has a specific permission."""
        return self.permissions.filter_by(code=permission_code).first() is not None
    
    def add_permission(self, permission: Permission):
        """Add permission to role."""
        if not self.has_permission(permission.code):
            self.permissions.append(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role."""
        if self.has_permission(permission.code):
            self.permissions.remove(permission)
    
    def __repr__(self):
        return f"<Role(code={self.code}, name={self.name})>"


# Association table for role-permission many-to-many relationship
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', Integer, ForeignKey('permissions.id'), primary_key=True),
    Column('assigned_at', DateTime(timezone=True), server_default=func.now())
)

# Back-populate the relationship
Permission.roles = relationship(
    "Role",
    secondary=role_permissions,
    back_populates="permissions",
    lazy="dynamic"
)


class User(BaseModel):
    """User model for authentication and authorization."""
    __tablename__ = 'users'
    
    # Basic Information
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20), nullable=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    
    # Authentication
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(32), nullable=False)
    
    # Status and Verification
    status = Column(Enum(UserStatus), default=UserStatus.PENDING_VERIFICATION, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    phone_verified = Column(Boolean, default=False, nullable=False)
    
    # Security
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    last_failed_login = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), server_default=func.now())
    must_change_password = Column(Boolean, default=False, nullable=False)
    
    # Tokens
    verification_token = Column(String(255), nullable=True, index=True)
    reset_token = Column(String(255), nullable=True, index=True)
    reset_token_expires = Column(DateTime(timezone=True), nullable=True)
    
    # Profile Information
    avatar_url = Column(String(500), nullable=True)
    department = Column(String(100), nullable=True)
    job_title = Column(String(100), nullable=True)
    employee_id = Column(String(50), nullable=True, index=True)
    
    # Territory and Location (for field users)
    territory_code = Column(String(20), nullable=True, index=True)
    division_code = Column(String(20), nullable=True, index=True)
    region_code = Column(String(20), nullable=True, index=True)
    home_latitude = Column(String(20), nullable=True)
    home_longitude = Column(String(20), nullable=True)
    
    # Preferences
    timezone = Column(String(50), default='UTC', nullable=False)
    language = Column(String(10), default='en', nullable=False)
    theme = Column(String(20), default='light', nullable=False)
    
    # System Fields
    last_activity = Column(DateTime(timezone=True), nullable=True)
    session_token = Column(String(255), nullable=True, index=True)
    
    # Relationships
    roles = relationship(
        "Role",
        secondary=user_roles,
        backref=backref("users", lazy="dynamic"),
        lazy="dynamic"
    )
    
    permissions = relationship(
        "Permission",
        secondary=user_permissions,
        backref=backref("users", lazy="dynamic"),
        lazy="dynamic"
    )
    
    # Foreign Key Relationships
    created_dealers = relationship("Dealer", foreign_keys="Dealer.created_by_user_id", back_populates="creator")
    managed_dealers = relationship("Dealer", foreign_keys="Dealer.manager_user_id", back_populates="manager")
    
    @hybrid_property
    def full_name(self):
        """Get full name."""
        return f"{self.first_name} {self.last_name}"
    
    @hybrid_property
    def is_active_user(self):
        """Check if user is active."""
        return self.status == UserStatus.ACTIVE and self.is_active
    
    @hybrid_property
    def is_locked(self):
        """Check if user account is locked."""
        return self.status == UserStatus.LOCKED or self.failed_login_attempts >= 5
    
    def set_password(self, password: str):
        """Set user password with salt and hash."""
        self.salt = secrets.token_hex(16)
        self.password_hash = generate_password_hash(password + self.salt)
        self.password_changed_at = datetime.utcnow()
        self.must_change_password = False
    
    def check_password(self, password: str) -> bool:
        """Check if provided password is correct."""
        return check_password_hash(self.password_hash, password + self.salt)
    
    def generate_verification_token(self) -> str:
        """Generate verification token."""
        self.verification_token = secrets.token_urlsafe(32)
        return self.verification_token
    
    def generate_reset_token(self, expires_hours: int = 24) -> str:
        """Generate password reset token."""
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expires = datetime.utcnow() + timedelta(hours=expires_hours)
        return self.reset_token
    
    def is_reset_token_valid(self, token: str) -> bool:
        """Check if reset token is valid."""
        return (
            self.reset_token == token and
            self.reset_token_expires and
            self.reset_token_expires > datetime.utcnow()
        )
    
    def record_login_attempt(self, success: bool):
        """Record login attempt."""
        if success:
            self.failed_login_attempts = 0
            self.last_login = datetime.utcnow()
            self.last_activity = datetime.utcnow()
            if self.status == UserStatus.LOCKED:
                self.status = UserStatus.ACTIVE
        else:
            self.failed_login_attempts += 1
            self.last_failed_login = datetime.utcnow()
            if self.failed_login_attempts >= 5:
                self.status = UserStatus.LOCKED
    
    def has_role(self, role_code: str) -> bool:
        """Check if user has a specific role."""
        return self.roles.filter_by(code=role_code).first() is not None
    
    def has_permission(self, permission_code: str) -> bool:
        """Check if user has a specific permission (direct or through roles)."""
        # Check direct permissions
        if self.permissions.filter_by(code=permission_code).first():
            return True
        
        # Check permissions through roles
        for role in self.roles:
            if role.has_permission(permission_code):
                return True
        
        return False
    
    def add_role(self, role: Role):
        """Add role to user."""
        if not self.has_role(role.code):
            self.roles.append(role)
    
    def remove_role(self, role: Role):
        """Remove role from user."""
        if self.has_role(role.code):
            self.roles.remove(role)
    
    def get_all_permissions(self) -> List[Permission]:
        """Get all permissions (direct + through roles)."""
        permissions = set()
        
        # Add direct permissions
        for perm in self.permissions:
            permissions.add(perm)
        
        # Add permissions through roles
        for role in self.roles:
            for perm in role.permissions:
                permissions.add(perm)
        
        return list(permissions)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def lock_account(self, reason: str = None):
        """Lock user account."""
        self.status = UserStatus.LOCKED
        if reason:
            self.update_metadata('lock_reason', reason)
    
    def unlock_account(self):
        """Unlock user account."""
        self.status = UserStatus.ACTIVE
        self.failed_login_attempts = 0
    
    def to_dict(self, include_sensitive: bool = False):
        """Convert to dictionary, optionally excluding sensitive data."""
        data = super().to_dict()
        
        if not include_sensitive:
            # Remove sensitive fields
            sensitive_fields = [
                'password_hash', 'salt', 'verification_token', 
                'reset_token', 'session_token'
            ]
            for field in sensitive_fields:
                data.pop(field, None)
        
        # Add computed fields
        data['full_name'] = self.full_name
        data['is_locked'] = self.is_locked
        data['role_codes'] = [role.code for role in self.roles]
        
        return data
    
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email}, status={self.status})>"


class UserSession(BaseModel):
    """User session tracking."""
    __tablename__ = 'user_sessions'
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    is_mobile = Column(Boolean, default=False)
    device_info = Column(Text, nullable=True)  # JSON string
    
    # Relationship
    user = relationship("User", backref="sessions")
    
    @hybrid_property
    def is_expired(self):
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_valid(self):
        """Check if session is valid."""
        return self.is_active and not self.is_expired
    
    def extend_session(self, hours: int = 8):
        """Extend session expiry."""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.last_activity = datetime.utcnow()
    
    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, token={self.session_token[:10]}...)>"


class UserLoginLog(BaseModel):
    """User login activity log."""
    __tablename__ = 'user_login_logs'
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    login_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    logout_time = Column(DateTime(timezone=True), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    success = Column(Boolean, nullable=False)
    failure_reason = Column(String(255), nullable=True)
    session_duration = Column(Integer, nullable=True)  # Duration in minutes
    
    # Relationship
    user = relationship("User", backref="login_logs")
    
    def calculate_session_duration(self):
        """Calculate session duration if logout time is available."""
        if self.logout_time and self.login_time:
            delta = self.logout_time - self.login_time
            self.session_duration = int(delta.total_seconds() // 60)
    
    def __repr__(self):
        return f"<UserLoginLog(user_id={self.user_id}, success={self.success}, time={self.login_time})>"