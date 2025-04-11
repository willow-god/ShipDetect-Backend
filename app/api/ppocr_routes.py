from fastapi import APIRouter, UploadFile, File, Form
from io import BytesIO
from PIL import Image
import numpy as np
from app.models.ppocr_model import PPOCRModel
import random

router = APIRouter()

@router.post("/detect_text_regions")
async def detect_text_regions(img_path: str = Form(..., description="The path to the image for OCR processing.")):
    """
    接受图片路径，检测图片中的文本区域并返回坐标。
    """
    ppo_cr_model = PPOCRModel()
    boxes = ppo_cr_model.detect_text_regions(img_path, save_crops=True)
    return {"boxes": boxes}

@router.post("/detect_text_regions_from_image")
async def detect_text_regions_from_image(file: UploadFile = File(..., description="The image file for OCR processing.")):
    """
    接受图片流，检测图片中的文本区域并返回坐标。
    """
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    img_array = np.array(img)
    
    ppo_cr_model = PPOCRModel()
    boxes = ppo_cr_model.detect_text_regions_from_image(img_array, save_crops=True)
    return {"boxes": boxes}

@router.post("/recognize_text")
async def recognize_text(img_path: str = Form(..., description="The path to the image for OCR recognition.")):
    """
    接受图片路径，识别图片中的文字内容。
    """
    ppo_cr_model = PPOCRModel()
    texts = ppo_cr_model.recognize_text(img_path)
    return {"texts": texts}

@router.post("/recognize_text_from_image")
async def recognize_text_from_image(file: UploadFile = File(..., description="The image file for OCR recognition.")):
    """
    接受图片流，识别图片中的文字内容。
    """
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    img_array = np.array(img)
    
    ppo_cr_model = PPOCRModel()
    texts = ppo_cr_model.recognize_text_from_image(img_array)
    return {"texts": texts}

# 模拟 PP-OCR 识别返回船牌号及 bbox
def simulate_ppocr(image_path):
    return {
        'ship_id': f"鄂A{random.randint(1000,9999)}",
        'ship_id_bbox': [10, 10, 120, 40]
    }