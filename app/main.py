# 入口文件
from fastapi import FastAPI
from app.api import sample_routes, yolov8_routes, ppocr_routes, video_routes, result_routes, ship_id_routes

app = FastAPI()

# 注册路由
app.include_router(sample_routes.router, prefix="/api/sample", tags=["Sample"])
# app.include_router(yolov8_routes.router, prefix="/api/yolov8", tags=["YOLOv8"])
# app.include_router(ppocr_routes.router, prefix="/api/ppocr", tags=["PP-OCR"])
app.include_router(video_routes.router, prefix="/api/video", tags=["Video Processing"])
app.include_router(result_routes.router, prefix="/api/result", tags=["Result"])
app.include_router(sample_routes.router, prefix="/api/picture", tags=["Picture Processing"])
app.include_router(ship_id_routes.router, prefix="/api/ship_id", tags=["Ship ID"])

# 中间件，防止跨域报错，允许所有来源
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 主页路由
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}
