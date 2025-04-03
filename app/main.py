from fastapi import FastAPI
from app.api.v1.router import router

app = FastAPI(
    title="ShipRecog API",
    description="船舶识别后端 API，提供基本接口。",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 注册 API 路由
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
