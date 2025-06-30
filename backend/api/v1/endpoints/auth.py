# # # api/v1/endpoints/auth.py
# # from datetime import datetime, timedelta
# # from typing import Any
# # from fastapi import APIRouter, Depends, HTTPException, status
# # from fastapi.security import OAuth2PasswordRequestForm
# # from sqlalchemy.orm import Session
# # from core.dependencies import get_db
# # from core.security import create_access_token, verify_password, get_password_hash, get_current_user
# # from models.user import User
# # from schemas.auth import Token, UserCreate, UserResponse, UserLogin

# # router = APIRouter()

# # @router.post("/register", response_model=UserResponse)
# # def register_user(user: UserCreate, db: Session = Depends(get_db)):
# #     """Register a new user"""
# #     # Check if user already exists
# #     db_user = db.query(User).filter(User.email == user.email).first()
# #     if db_user:
# #         raise HTTPException(
# #             status_code=400,
# #             detail="Email already registered"
# #         )
    
# #     # Create new user
# #     hashed_password = get_password_hash(user.password)
# #     db_user = User(
# #         email=user.email,
# #         username=user.username,
# #         hashed_password=hashed_password,
# #         full_name=user.full_name,
# #         is_active=True
# #     )
# #     db.add(db_user)
# #     db.commit()
# #     db.refresh(db_user)
    
# #     return db_user

# # @router.post("/login", response_model=Token)
# # def login_for_access_token(
# #     form_data: OAuth2PasswordRequestForm = Depends(),
# #     db: Session = Depends(get_db)
# # ):
# #     """Authenticate user and return access token"""
# #     user = db.query(User).filter(User.email == form_data.username).first()
    
# #     if not user or not verify_password(form_data.password, user.hashed_password):
# #         raise HTTPException(
# #             status_code=status.HTTP_401_UNAUTHORIZED,
# #             detail="Incorrect email or password",
# #             headers={"WWW-Authenticate": "Bearer"},
# #         )
    
# #     access_token_expires = timedelta(minutes=30)
# #     access_token = create_access_token(
# #         data={"sub": user.email}, expires_delta=access_token_expires
# #     )
    
# #     return {"access_token": access_token, "token_type": "bearer"}

# # @router.post("/refresh", response_model=Token)
# # def refresh_token(current_user: User = Depends(get_current_user)):
# #     """Refresh access token"""
# #     access_token_expires = timedelta(minutes=30)
# #     access_token = create_access_token(
# #         data={"sub": current_user.email}, expires_delta=access_token_expires
# #     )
# #     return {"access_token": access_token, "token_type": "bearer"}

# # @router.post("/logout")
# # def logout(current_user: User = Depends(get_current_user)):
# #     """Logout user (invalidate token)"""
# #     # In a real app, you might want to blacklist the token
# #     return {"message": "Successfully logged out"}

# # @router.get("/me", response_model=UserResponse)
# # def read_users_me(current_user: User = Depends(get_current_user)):
# #     """Get current user info"""
# #     return current_user

# # api/v1/endpoints/auth.py
# from datetime import datetime, timedelta
# from typing import Optional
# from fastapi import APIRouter, Depends, HTTPException, status, Form, BackgroundTasks
# from fastapi.security import OAuth2PasswordRequestForm
# from sqlalchemy.orm import Session
# from jose import JWTError, jwt
# from passlib.context import CryptContext

# from core.dependencies import get_db
# from core.security import create_access_token, create_refresh_token, verify_token, get_current_user
# from config.settings import settings
# from models.user import User
# from schemas.auth import Token, UserCreate, UserResponse, UserLogin, RefreshToken
# from schemas.base import APIResponse

# router = APIRouter(prefix="/auth", tags=["Authentication"])

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# @router.post("/register", response_model=APIResponse[UserResponse])
# async def register_user(
#     user_data: UserCreate,
#     db: Session = Depends(get_db)
# ):
#     """
#     Register a new user (dealer/admin)
#     """
#     try:
#         # Check if user already exists
#         existing_user = db.query(User).filter(
#             (User.email == user_data.email) | (User.username == user_data.username)
#         ).first()
        
#         if existing_user:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="User with this email or username already exists"
#             )
        
#         # Hash password
#         hashed_password = pwd_context.hash(user_data.password)
        
#         # Create new user
#         new_user = User(
#             username=user_data.username,
#             email=user_data.email,
#             full_name=user_data.full_name,
#             hashed_password=hashed_password,
#             role=user_data.role,
#             division_code=user_data.division_code,
#             territory_code=user_data.territory_code,
#             is_active=True,
#             created_at=datetime.utcnow()
#         )
        
#         db.add(new_user)
#         db.commit()
#         db.refresh(new_user)
        
#         return APIResponse(
#             success=True,
#             message="User registered successfully",
#             data=UserResponse.from_orm(new_user)
#         )
        
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Registration failed: {str(e)}"
#         )

# @router.post("/login", response_model=APIResponse[Token])
# async def login(
#     form_data: OAuth2PasswordRequestForm = Depends(),
#     db: Session = Depends(get_db)
# ):
#     """
#     Login endpoint for dealers and admins
#     """
#     try:
#         # Find user by username or email
#         user = db.query(User).filter(
#             (User.username == form_data.username) | (User.email == form_data.username)
#         ).first()
        
#         if not user or not pwd_context.verify(form_data.password, user.hashed_password):
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Incorrect username or password",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )
        
#         if not user.is_active:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="User account is disabled"
#             )
        
#         # Update last login
#         user.last_login = datetime.utcnow()
#         db.commit()
        
#         # Create tokens
#         access_token = create_access_token(
#             data={"sub": user.username, "user_id": user.id, "role": user.role}
#         )
#         refresh_token = create_refresh_token(
#             data={"sub": user.username, "user_id": user.id}
#         )
        
#         token_data = Token(
#             access_token=access_token,
#             refresh_token=refresh_token,
#             token_type="bearer",
#             expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
#             user=UserResponse.from_orm(user)
#         )
        
#         return APIResponse(
#             success=True,
#             message="Login successful",
#             data=token_data
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Login failed: {str(e)}"
#         )

# @router.post("/refresh", response_model=APIResponse[Token])
# async def refresh_access_token(
#     refresh_data: RefreshToken,
#     db: Session = Depends(get_db)
# ):
#     """
#     Refresh access token using refresh token
#     """
#     try:
#         # Verify refresh token
#         payload = verify_token(refresh_data.refresh_token)
#         username = payload.get("sub")
#         user_id = payload.get("user_id")
        
#         if not username or not user_id:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Invalid refresh token"
#             )
        
#         # Get user
#         user = db.query(User).filter(User.id == user_id).first()
#         if not user or not user.is_active:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="User not found or inactive"
#             )
        
#         # Create new access token
#         access_token = create_access_token(
#             data={"sub": user.username, "user_id": user.id, "role": user.role}
#         )
        
#         token_data = Token(
#             access_token=access_token,
#             refresh_token=refresh_data.refresh_token,
#             token_type="bearer",
#             expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
#             user=UserResponse.from_orm(user)
#         )
        
#         return APIResponse(
#             success=True,
#             message="Token refreshed successfully",
#             data=token_data
#         )
        
#     except JWTError:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid refresh token"
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Token refresh failed: {str(e)}"
#         )

# @router.post("/logout", response_model=APIResponse[dict])
# async def logout(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Logout user (token blacklisting would be implemented here)
#     """
#     try:
#         # Update last logout time
#         current_user.last_logout = datetime.utcnow()
#         db.commit()
        
#         return APIResponse(
#             success=True,
#             message="Logout successful",
#             data={"message": "Successfully logged out"}
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Logout failed: {str(e)}"
#         )

# @router.get("/me", response_model=APIResponse[UserResponse])
# async def get_current_user_info(
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Get current user information
#     """
#     return APIResponse(
#         success=True,
#         message="User information retrieved successfully",
#         data=UserResponse.from_orm(current_user)
#     )

# @router.post("/change-password", response_model=APIResponse[dict])
# async def change_password(
#     current_password: str = Form(...),
#     new_password: str = Form(...),
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Change user password
#     """
#     try:
#         # Verify current password
#         if not pwd_context.verify(current_password, current_user.hashed_password):
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Current password is incorrect"
#             )
        
#         # Hash new password
#         hashed_new_password = pwd_context.hash(new_password)
        
#         # Update password
#         current_user.hashed_password = hashed_new_password
#         current_user.updated_at = datetime.utcnow()
#         db.commit()
        
#         return APIResponse(
#             success=True,
#             message="Password changed successfully",
#             data={"message": "Password updated successfully"}
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Password change failed: {str(e)}"
#         )

# @router.post("/reset-password-request", response_model=APIResponse[dict])
# async def request_password_reset(
#     email: str = Form(...),
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db)
# ):
#     """
#     Request password reset (sends email with reset token)
#     """
#     try:
#         user = db.query(User).filter(User.email == email).first()
        
#         if not user:
#             # Don't reveal if email exists or not
#             return APIResponse(
#                 success=True,
#                 message="If email exists, password reset instructions have been sent",
#                 data={"message": "Reset instructions sent"}
#             )
        
#         # Generate reset token
#         reset_token = create_access_token(
#             data={"sub": user.username, "user_id": user.id, "type": "password_reset"},
#             expires_delta=timedelta(minutes=30)
#         )
        
#         # Store reset token in database
#         user.reset_token = reset_token
#         user.reset_token_expires = datetime.utcnow() + timedelta(minutes=30)
#         db.commit()
        
#         # Add background task to send email
#         # background_tasks.add_task(send_password_reset_email, user.email, reset_token)
        
#         return APIResponse(
#             success=True,
#             message="Password reset instructions sent to email",
#             data={"message": "Reset instructions sent"}
#         )
        
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Password reset request failed: {str(e)}"
#         )

# @router.post("/reset-password", response_model=APIResponse[dict])
# async def reset_password(
#     token: str = Form(...),
#     new_password: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     """
#     Reset password using reset token
#     """
#     try:
#         # Verify reset token
#         payload = verify_token(token)
#         user_id = payload.get("user_id")
#         token_type = payload.get("type")
        
#         if not user_id or token_type != "password_reset":
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid reset token"
#             )
        
#         # Get user and verify token
#         user = db.query(User).filter(User.id == user_id).first()
#         if not user or user.reset_token != token:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid or expired reset token"
#             )
        
#         if user.reset_token_expires < datetime.utcnow():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Reset token has expired"
#             )
        
#         # Hash new password and update
#         hashed_password = pwd_context.hash(new_password)
#         user.hashed_password = hashed_password
#         user.reset_token = None
#         user.reset_token_expires = None
#         user.updated_at = datetime.utcnow()
#         db.commit()
        
#         return APIResponse(
#             success=True,
#             message="Password reset successfully",
#             data={"message": "Password has been reset"}
#         )
        
#     except HTTPException:
#         raise
#     except JWTError:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Invalid reset token"
#         )
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Password reset failed: {str(e)}"
#         )

from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from core.dependencies import get_db
from core.security import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_password_hash,
    verify_token
)
from models.user import User
from schemas.auth import (
    Token,
    TokenData,
    UserCreate,
    UserResponse,
    RefreshToken,
    PasswordReset,
    UserLogin
)
from services.auth_service import AuthService
from config.settings import settings

router = APIRouter()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user
    """
    auth_service = AuthService(db)
    
    # Check if user already exists
    if auth_service.get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    if auth_service.get_user_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    user = auth_service.create_user(user_data)
    return UserResponse.from_orm(user)

@router.post("/login", response_model=Token)
async def login(
    user_credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    User login endpoint
    """
    auth_service = AuthService(db)
    
    # Authenticate user
    user = auth_service.authenticate_user(
        user_credentials.username, 
        user_credentials.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access and refresh tokens
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.username, "user_id": user.id}
    )
    
    # Store refresh token in database
    auth_service.store_refresh_token(user.id, refresh_token)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )

@router.post("/login/form", response_model=Token)
async def login_form(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 compatible login endpoint
    """
    auth_service = AuthService(db)
    
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.username, "user_id": user.id}
    )
    
    auth_service.store_refresh_token(user.id, refresh_token)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_data: RefreshToken,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token
    """
    auth_service = AuthService(db)
    
    try:
        # Verify refresh token
        token_data = verify_token(refresh_data.refresh_token)
        user = auth_service.get_user_by_username(token_data.username)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Verify refresh token exists in database
        if not auth_service.verify_refresh_token(user.id, refresh_data.refresh_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new access token
        new_access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id}
        )
        
        return Token(
            access_token=new_access_token,
            refresh_token=refresh_data.refresh_token,
            token_type="bearer"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@router.post("/logout")
async def logout(
    refresh_data: RefreshToken,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout user and invalidate refresh token
    """
    auth_service = AuthService(db)
    
    # Remove refresh token from database
    auth_service.revoke_refresh_token(current_user.id, refresh_data.refresh_token)
    
    return {"message": "Successfully logged out"}

@router.post("/logout-all")
async def logout_all(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout from all devices (revoke all refresh tokens)
    """
    auth_service = AuthService(db)
    
    # Remove all refresh tokens for user
    auth_service.revoke_all_refresh_tokens(current_user.id)
    
    return {"message": "Successfully logged out from all devices"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information
    """
    return UserResponse.from_orm(current_user)

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update current user information
    """
    auth_service = AuthService(db)
    
    # Update user
    updated_user = auth_service.update_user(current_user.id, user_update)
    
    return UserResponse.from_orm(updated_user)

@router.post("/password-reset-request")
async def request_password_reset(
    email: str,
    db: Session = Depends(get_db)
):
    """
    Request password reset (sends email with reset token)
    """
    auth_service = AuthService(db)
    
    # Generate password reset token
    user = auth_service.get_user_by_email(email)
    if user:
        reset_token = auth_service.create_password_reset_token(user.id)
        # TODO: Send email with reset token
        # await send_password_reset_email(user.email, reset_token)
    
    # Always return success to prevent email enumeration
    return {"message": "If the email exists, a password reset link has been sent"}

@router.post("/password-reset")
async def reset_password(
    reset_data: PasswordReset,
    db: Session = Depends(get_db)
):
    """
    Reset password using reset token
    """
    auth_service = AuthService(db)
    
    # Verify reset token and update password
    if not auth_service.reset_password(reset_data.token, reset_data.new_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    return {"message": "Password successfully reset"}

@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user password
    """
    auth_service = AuthService(db)
    
    # Verify current password
    if not auth_service.verify_password(current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    auth_service.update_password(current_user.id, new_password)
    
    return {"message": "Password successfully changed"}

@router.get("/verify-token")
async def verify_access_token(
    current_user: User = Depends(get_current_user)
):
    """
    Verify if access token is valid
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username
    }