from fastapi import APIRouter, UploadFile, File, Form
from io import BytesIO
from PIL import Image
import numpy as np
import random
from app.models.yolov8_model import YOLOv8Model

router = APIRouter()

@router.post("/predict")
async def predict(img_path: str = Form(..., description="The path to the image for object detection.")):
    """
    接受图片路径，进行目标检测并返回结果。
    """
    yolo_model = YOLOv8Model(weights_path="your_model_path", config_path="your_config_path")
    results = yolo_model.predict(img_path)
    return {"results": results}

@router.post("/predict_from_image")
async def predict_from_image(file: UploadFile = File(..., description="The image file for object detection.")):
    """
    接受图片流，进行目标检测并返回结果。
    """
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    img_array = np.array(img)
    
    yolo_model = YOLOv8Model(weights_path="your_model_path", config_path="your_config_path")
    results = yolo_model.predict_image(img_array)
    return {"results": results}

# 模拟 YOLOv8 检测（返回多个结果）
def simulate_yolov8_detect(frame):
    results = []
    for _ in range(random.randint(1, 3)):  # 每帧1~3个目标
        x1, y1 = random.randint(0, 100), random.randint(0, 100)
        x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(30, 100)
        results.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': round(random.uniform(0.6, 0.95), 2),
            'category_id': random.randint(1, 6),
            'category': random.choice(['ore carrier', 'bulk cargo carrier', 'general cargo ship', 'container ship', 'fishing boat', 'passenger ship']),
        })
    return results

