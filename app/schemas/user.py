from pydantic import BaseModel

class UserBase(BaseModel):
    username: str
    role: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    disabled: bool = False
    
    class Config:
        orm_mode = True