from fastapi import APIRouter, UploadFile, File, Form
from io import BytesIO
from PIL import Image
import numpy as np
import tempfile
import os
from app.models.ppocr_model import PPOCRModel
from typing import Union
import random

# router = APIRouter()

# @router.post("/detect_text_regions")
# async def detect_text_regions(img_path: str = Form(..., description="The path to the image for OCR processing.")):
#     """
#     接受图片路径，检测图片中的文本区域并返回坐标。
#     """
#     ppo_cr_model = PPOCRModel()
#     boxes = ppo_cr_model.detect_text_regions(img_path, save_crops=True)
#     return {"boxes": boxes}

# @router.post("/detect_text_regions_from_image")
# async def detect_text_regions_from_image(file: UploadFile = File(..., description="The image file for OCR processing.")):
#     """
#     接受图片流，检测图片中的文本区域并返回坐标。
#     """
#     contents = await file.read()
#     img = Image.open(BytesIO(contents))
#     img_array = np.array(img)
    
#     ppo_cr_model = PPOCRModel()
#     boxes = ppo_cr_model.detect_text_regions_from_image(img_array, save_crops=True)
#     return {"boxes": boxes}

# @router.post("/recognize_text")
# async def recognize_text(img_path: str = Form(..., description="The path to the image for OCR recognition.")):
#     """
#     接受图片路径，识别图片中的文字内容。
#     """
#     ppo_cr_model = PPOCRModel()
#     texts = ppo_cr_model.recognize_text(img_path)
#     return {"texts": texts}

# @router.post("/recognize_text_from_image")
# async def recognize_text_from_image(file: UploadFile = File(..., description="The image file for OCR recognition.")):
#     """
#     接受图片流，识别图片中的文字内容。
#     """
#     contents = await file.read()
#     img = Image.open(BytesIO(contents))
#     img_array = np.array(img)
    
#     ppo_cr_model = PPOCRModel()
#     texts = ppo_cr_model.recognize_text_from_image(img_array)
#     return {"texts": texts}

# 模拟 PP-OCR 识别返回船牌号及 bbox
def simulate_ppocr(image_path):
    return {
        'ship_id': f"鄂A{random.randint(1000,9999)}",
        'ship_id_bbox': [10, 10, 120, 40],
        'confidence': 0.95
    }

from paddleocr import PaddleOCR
import numpy as np

# 初始化 PaddleOCR（建议全局只初始化一次）
ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')

def ppocr_v4(image: Union[str, bytes, np.ndarray]):
    """
    支持文件路径（str）或图像字节流（bytes/np.ndarray）的 OCR 检测。
    返回格式：
    {
        'ship_id': str,
        'ship_id_bbox': [x1, y1, x2, y2],
        'confidence': float
    }
    """
    # 如果是字符串路径
    print(f"开始识别, 类型: {type(image)}")
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path does not exist: {image}")
        result = ocr_model.ocr(image, cls=True)

    # 如果是字节流或 numpy array
    else:
        # 如果是 bytes，转成 numpy
        if isinstance(image, bytes):
            import cv2
            np_arr = np.frombuffer(image, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            raise ValueError("Unsupported image input type.")

        # 写入本地文件（OCR 只支持路径）
        output_dir = "output/temp"
        os.makedirs(output_dir, exist_ok=True)
        temp_image_path = os.path.join(output_dir, "ppocr_v4.jpg")
        from PIL import Image
        pil_image = Image.fromarray(image_np[..., ::-1])  # BGR to RGB
        pil_image.save(temp_image_path)
        result = ocr_model.ocr(temp_image_path, cls=True)

    # 无结果或结果为空
    if not result or not result[0]:
        print("无法识别到船号")
        return {
            'ship_id': "无法检测",
            'ship_id_bbox': [0, 0, 0, 0],
            'confidence': 0.0
        }

    # 选置信度最高的结果
    best_line = max(result[0], key=lambda x: x[1][1] if x[1][1] is not None else 0)
    box_points = best_line[0]
    text = best_line[1][0]
    confidence = best_line[1][1] or 0.0

    # 提取矩形框
    x_coords = [point[0] for point in box_points]
    y_coords = [point[1] for point in box_points]
    bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]

    return {
        'ship_id': text,
        'ship_id_bbox': bbox,
        'confidence': round(confidence, 3)
    }