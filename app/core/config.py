from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router
from app.core.config import settings

app = FastAPI(title="Ship Detection API")

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含路由
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Ship Detection API is running"}