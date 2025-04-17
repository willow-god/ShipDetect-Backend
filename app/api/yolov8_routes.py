import cv2
from fastapi import APIRouter, UploadFile, File, Form
from io import BytesIO
from PIL import Image
import numpy as np
import random
from typing import List, Dict, Union
import os
from app.models.yolov8_model import YOLOv8Model

# router = APIRouter()

# 模型路径配置
weights_path = "E:/Graduate_Design/ShipDetect-Backend/resources/models/yolov8_m_50.pdparams"
config_path = "E:/Graduate_Design/ShipDetect-Backend/configs/default_config.yaml"

model = YOLOv8Model(
    weights_path=weights_path,
    config_path=config_path,
    use_gpu=True,
    threshold=0.3
)

# @router.post("/predict")
# async def predict(img_path: str = Form(..., description="The path to the image for object detection.")):
#     """
#     接受图片路径，进行目标检测并返回结果。
#     """
#     yolo_model = YOLOv8Model(weights_path="your_model_path", config_path="your_config_path")
#     results = yolo_model.predict(img_path)
#     return {"results": results}

# @router.post("/predict_from_image")
# async def predict_from_image(file: UploadFile = File(..., description="The image file for object detection.")):
#     """
#     接受图片流，进行目标检测并返回结果。
#     """
#     contents = await file.read()
#     img = Image.open(BytesIO(contents))
#     img_array = np.array(img)
    
#     yolo_model = YOLOv8Model(weights_path="your_model_path", config_path="your_config_path")
#     results = yolo_model.predict_image(img_array)
#     return {"results": results}

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

def yolov8_detect(frame: Union[str, np.ndarray, Image.Image]) -> List[Dict]:
    """
    调用 YOLOv8 进行检测，并统一返回与 simulate_yolov8_detect 相同格式的结果。

    返回字段：
    - bbox: [x1, y1, x2, y2]
    - confidence: float
    - category_id: int
    - category: str
    """
    # 预测
    print(f"开始检测, 类型: {type(frame)}")
    if isinstance(frame, str):
        results = model.predict(frame, threshold=0.3)  # 直接传入文件路径进行预测
    else:
        # 写入到一个文件中，然后再通过路径进行检测

        temp = "./output/temp/yolov8_detect.jpg"
        if isinstance(frame, np.ndarray):
            cv2.imwrite(temp, frame)
        elif isinstance(frame, Image.Image):
            frame.save(temp)
        else:
            raise ValueError("Unsupported image type.")
        # 预测
        results = model.predict(temp, threshold=0.3)
        # results = model.predict_image(frame, threshold=0.3)
    
    print(f"检测完成, 结果: {results}")

    # 筛选逻辑
    high_conf = [r for r in results if r["score"] >= 0.5]
    if high_conf:
        selected = high_conf
    else:
        mid_conf = [r for r in results if r["score"] >= 0.3]
        if mid_conf:
            selected = [max(mid_conf, key=lambda r: r["score"])]
        else:
            return []

    # 转换字段名为模拟输出格式
    formatted = []
    for r in selected:
        formatted.append({
            "bbox": r["bbox"],
            "confidence": round(r["score"], 2),
            "category_id": r["category_id"] + 1,  # 如果你的category_id原本从0开始，可以+1以匹配模拟逻辑
            "category": r["category"],
        })

    return formatted