from app.models.yolov8_model import YOLOv8Model
import os

# 获取当前文件的绝对路径
# current_file_path = os.path.abspath(__file__)
# weights_path = os.path.join(os.path.dirname(current_file_path), 'resources/models/yolov8_m_50_epoches.pdparams')
# default_config_path = os.path.join(os.path.dirname(current_file_path), 'configs/default_config.yaml')
# image_path = os.path.join(os.path.dirname(current_file_path), 'resources/images/000001.jpg')

# # test yolov8_model.py
# model = YOLOv8Model(weights_path=weights_path, config_path=default_config_path, use_gpu=True, threshold=0.3)

# results = model.predict(image_path, threshold=0.45, save_result=True)
# print("Results:", results)

# test ppocr_model.py
from app.models.ppocr_model import PPOCRModel

PPOCR = PPOCRModel()
image_path = os.path.join(os.path.dirname(__file__), 'resources/images/000008.jpg')
results_det = PPOCR.detect_text_regions(image_path, save_crops=True)
print("Detection Results:", results_det)