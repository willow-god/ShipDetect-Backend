import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from PIL import Image
import numpy as np
import io
import cv2
import tempfile
from datetime import datetime
from app.models.yolov8_model import YOLOv8Model
from app.core.config import settings
from app.core.security import get_current_user  # 假设后期会有用户认证

router = APIRouter()

# 初始化模型
model = YOLOv8Model(
    weights_path=settings.YOLO_WEIGHTS_PATH,
    config_path=settings.YOLO_CONFIG_PATH
)

# 确保输出目录存在
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

async def save_upload_file(upload_file: UploadFile) -> str:
    """保存上传文件到临时位置"""
    try:
        # 创建带有时间戳的临时文件
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = os.path.splitext(upload_file.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, 
            prefix=f"upload_{timestamp}_",
            delete=False
        )
        contents = await upload_file.read()
        temp_file.write(contents)
        return temp_file.name
    finally:
        await upload_file.close()

@router.post("/image", summary="上传图片并返回处理后的图片", response_class=FileResponse)
async def predict_and_return_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = 0.5,
    # current_user: dict = Depends(get_current_user)  # 后期添加用户认证
):
    """
    上传图片，返回带有检测框的图片
    
    - **file**: 上传的图片文件
    - **threshold**: 检测阈值(0-1)
    """
    try:
        # 保存上传文件
        temp_path = await save_upload_file(file)
        
        # 执行推理并保存结果图片
        output_filename = f"result_{os.path.basename(temp_path)}"
        output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
        
        # 执行推理，保存可视化结果
        model.predict(
            image_path=temp_path,
            threshold=threshold,
            save_result=True,
            output_dir=settings.OUTPUT_DIR
        )
        
        # 返回处理后的图片
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="结果图片生成失败")
        
        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename=output_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/data", summary="上传图片并返回检测数据", response_class=JSONResponse)
async def predict_and_return_data(
    file: UploadFile = File(...),
    threshold: Optional[float] = 0.5,
    # current_user: dict = Depends(get_current_user)  # 后期添加用户认证
):
    """
    上传图片，返回检测结果的JSON数据
    
    - **file**: 上传的图片文件
    - **threshold**: 检测阈值(0-1)
    """
    try:
        # 将上传文件转换为PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 执行推理
        results = model.predict_image(
            image=image,
            threshold=threshold,
            save_result=False
        )
        
        return {
            "status": "success",
            "data": results,
            "image_size": image.size,
            "detection_count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

# 后期可以添加需要认证的接口
# @router.post("/secure/predict")
# async def secure_predict(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
#     ...