from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.core.security import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.schemas.token import Token  # 需要创建对应的Pydantic模型
from app.schemas.user import User    # 需要创建对应的Pydantic模型

router = APIRouter()

@router.post("/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    OAuth2兼容的令牌获取接口，返回access_token
    
    - **username**: 用户名
    - **password**: 密码
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=User, tags=["Authentication"])
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
):
    """获取当前登录用户信息"""
    return current_user

# 在auth.py中添加
from fastapi import Security
from app.core.security import check_permission

@router.get("/admin-only", tags=["Authentication"])
async def admin_route(
    user: User = Security(check_permission("admin")),  # 需要admin权限
):
    return {"message": "管理员专属区域"}