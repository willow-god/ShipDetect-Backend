@echo off
:: 设置虚拟环境路径，如果你使用了虚拟环境的话
conda activate GD-PP

:: 设置环境变量
set FASTAPI_HOST=0.0.0.0
set FASTAPI_PORT=8000

:: 启动FastAPI服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload