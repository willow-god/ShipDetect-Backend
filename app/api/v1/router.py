from fastapi import APIRouter
from app.api.v1.endpoints import auth, yolov8, health  # 添加auth导入

router = APIRouter()

# 取消auth路由的注释
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(yolov8.router, prefix="/predict/yolov8", tags=["YOLOv8"])
router.include_router(health.router, prefix="/health", tags=["Health Check"])