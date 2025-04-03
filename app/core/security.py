from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ----------------------------
# 配置和模拟用户数据
# ----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

SECRET_KEY = "your-secret-key-for-dev"  # 生产环境应从环境变量获取
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 模拟用户数据库
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "permissions": ["all"],
        "disabled": False
    },
    "user1": {
        "username": "user1",
        "hashed_password": pwd_context.hash("user123"),
        "role": "user",
        "permissions": ["view", "upload"],
        "disabled": False
    }
}

# ----------------------------
# Pydantic模型
# ----------------------------
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    role: str
    permissions: List[str]
    disabled: bool

class UserInDB(User):
    hashed_password: str

# ----------------------------
# 核心功能实现
# ----------------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str) -> Optional[UserInDB]:
    """从数据库获取用户"""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """认证用户"""
    user = get_user(fake_users_db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建JWT访问令牌
    
    Args:
        data: 需要编码的数据 (e.g. {"sub": username})
        expires_delta: 过期时间差
        
    Returns:
        str: JWT令牌
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    从JWT令牌获取当前用户
    
    Raises:
        HTTPException: 如果令牌无效或用户不存在
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    获取当前活跃用户（检查用户是否被禁用）
    
    Raises:
        HTTPException: 如果用户被禁用
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="用户已被禁用")
    return current_user

def check_permission(required_permission: str):
    """
    权限检查依赖项工厂函数
    
    Args:
        required_permission: 需要的权限字符串
        
    Returns:
        Depends: FastAPI依赖项
    """
    def permission_checker(current_user: User = Depends(get_current_active_user)):
        if "all" in current_user.permissions:
            return current_user
        if required_permission not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要 {required_permission} 权限"
            )
        return current_user
    return permission_checker