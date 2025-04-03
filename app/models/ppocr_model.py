from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import cv2
import os

class PPOCRModel:
    def __init__(self, lang='ch', use_angle_cls=True, show_log=False):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=show_log)  # need to run only once to download and load model into memory
        self.results = []  # 存储识别结果

    def detect_text_regions(self, img_path, save_crops=False, crop_dir='./output/PPOOCRv4'):
        """
        检测文本区域
        :param img_path: 图片路径
        :param conf_threshold: 置信度阈值
        :param save_crops: 是否保存裁剪的区域
        :param crop_dir: 保存裁剪图片的目录
        :return: 检测到的文本区域（bbox 坐标列表）
        """
        # 检查图片路径是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image path {img_path} does not exist.")
        result = self.ocr.ocr(img_path, rec=False)
        boxes = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                boxes.append(line)  # 获取文本区域的坐标
                # 每个坐标分别为 [左上角, 右上角, 右下角, 左下角]，每个坐标为 [x, y] 格式
        
        if save_crops:
            # 创建目录
            if not os.path.exists(crop_dir):
                os.makedirs(crop_dir)
            image = cv2.imread(img_path)
            # 遍历检测到的文本区域，并保存裁剪的图片
            for box in boxes:
                # 按照顺序提取左上角、右上角、右下角、左下角的坐标
                left_top = box[0]
                right_top = box[1]
                right_bottom = box[2]
                left_bottom = box[3]
                # 将坐标转换为整数
                points = np.array([left_top, right_top, right_bottom, left_bottom], dtype=np.int32)
                # 创建掩膜
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [points], (255, 255, 255))
                # 裁剪图片
                crop = cv2.bitwise_and(image, mask)
                # 获取裁剪区域的最小外接矩形
                x, y, w, h = cv2.boundingRect(points)
                crop = crop[y:y+h, x:x+w]
                # 保存裁剪的图片
                crop_name = os.path.join(crop_dir, f'crop_{idx}.jpg')
                cv2.imwrite(crop_name, crop)
                idx += 1
                print(f"Saved cropped image: {crop_name}")
        return boxes

    def recognize_text(self, img_path, conf_threshold=0.5):
        """
        识别图片中的文字
        :param img_path: 图片路径
        :param conf_threshold: 置信度阈值
        :return: 文本内容列表及对应置信度
        """
        result = self.ocr.ocr(img_path, det=False, cls=False)
        texts = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                if line[1] >= conf_threshold:
                    texts.append(line)
    
        return texts

    def detect_and_recognize(self, img_path, conf_threshold=0.5, visualize=False, save_results=None, font_path='static/simfang.ttf'):
        """
        先检测再识别
        :param img_path: 图片路径
        :param conf_threshold: 置信度阈值
        :param visualize: 是否可视化
        :param save_results: 结果保存路径（文本文件）
        :param font_path: 字体路径
        :return: 识别结果
        """
        result = self.ocr.ocr(img_path, cls=True, det=True)
        bboxes = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                if line[1][1] >= conf_threshold:
                    boxes.append(line)

                    
        # image = Image.open(img_path).convert('RGB')
        
        # boxes, texts, scores = [], [], []
        # for line in result:
        #     if line[1][1] >= conf_threshold:
        #         boxes.append(line[0])
        #         texts.append(line[1][0])
        #         scores.append(line[1][1])
        
        # self.results = texts  # 存储识别文本
        
        # if visualize:
        #     im_show = draw_ocr(image, boxes, texts, scores, font_path=font_path)
        #     im_show = Image.fromarray(im_show)
        #     im_show.show()
        
        # if save_results:
        #     with open(save_results, 'w', encoding='utf-8') as f:
        #         for text, score in zip(texts, scores):
        #             f.write(f'{text}: {score}\n')
        
        return bboxes

    def fuzzy_match(self, keyword):
        """
        模糊匹配，提高正确率
        :param keyword: 需要匹配的关键字
        :return: 匹配到的文本
        """
        matches = [text for text in self.results if keyword in text]
        return matches
