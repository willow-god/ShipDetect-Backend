import tempfile
import os
from PIL import Image
import cv2
import numpy as np
import warnings
from typing import List, Dict, Optional, Union
import paddle
from ppdet.core.workspace import load_config, AttrDict
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_version, check_config

warnings.filterwarnings('ignore')

class YOLOv8Model:
    def __init__(
        self,
        weights_path: str,
        config_path: str,
        use_gpu: bool = True,
        threshold: float = 0.5,
        device_id: int = 0
    ):
        """
        初始化 YOLOv8 模型
        
        Args:
            weights_path (str): 模型权重文件路径 (.pdparams)
            config_path (str): 配置文件路径
            use_gpu (bool): 是否使用GPU
            threshold (float): 默认检测阈值
            device_id (int): GPU设备ID
        """
        # 设置设备
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.threshold = threshold
        
        # 初始化配置
        self.cfg = self._init_config(config_path, weights_path)
        
        # 初始化训练器（用于推理）
        print("Initializing Trainer...")
        self.trainer = Trainer(self.cfg, mode='test')
        self.trainer.load_weights(self.cfg.weights)
        
    def _init_config(self, config_path: str, weights_path: str) -> AttrDict:
        """初始化配置"""        
        cfg = load_config(config_path)
        cfg.weights = weights_path
        cfg.use_gpu = self.use_gpu
        
        if self.use_gpu:
            paddle.set_device(f'gpu:{self.device_id}')
        else:
            paddle.set_device('cpu')
        
        cfg.draw_threshold = self.threshold
        cfg.infer_dir = None
        cfg.infer_img = None
        cfg.output_dir = 'output'
        cfg.visualize = False
        
        check_config(cfg)
        check_gpu(cfg.use_gpu)
        check_version()
        
        return cfg
    
    def predict(
        self,
        image_path: str,
        threshold: Optional[float] = None,
        save_result: bool = False,
        output_dir: Optional[str] = None, 
        slice_infer: bool = False,
    ) -> List[Dict]:
        """
        对单张图片进行推理（使用文件路径）
        
        Args:
            image_path (str): 图片路径
            threshold (float, optional): 检测阈值，如果为None则使用初始化时的阈值
            save_result (bool): 是否保存可视化结果
            output_dir (str, optional): 结果保存目录
            slice_infer (bool): 是否使用切片推理
            
        Returns:
            List[Dict]: 检测结果列表
        """
        # 设置阈值
        draw_threshold = threshold if threshold is not None else self.threshold
        
        # 设置输出目录
        if output_dir is not None:
            self.cfg.output_dir = output_dir
        
        # 设置是否保存结果
        self.cfg.visualize = save_result
        
        # 执行推理
        if slice_infer:
            results = self.trainer.slice_predict(
                [image_path],
                slice_size=(640, 640),
                overlap_ratio=[0.25, 0.25],
                combine_method='nms',
                match_threshold=0.6,
                match_metric='ios', 
                draw_threshold=draw_threshold,
                output_dir=self.cfg.output_dir,
                save_results=save_result,
                visualize=save_result
            )
        else:
            results = self.trainer.predict(
                [image_path],
                draw_threshold=draw_threshold,
                output_dir=self.cfg.output_dir,
                save_results=save_result,
                visualize=save_result
            )
        
        return self._format_results(results, threshold)
    
    def predict_image(
        self,
        image: Union[np.ndarray, Image.Image],
        threshold: Optional[float] = None,
        save_result: bool = False,
        output_dir: Optional[str] = None, 
        slice_infer: bool = False,
    ) -> List[Dict]:
        """
        对图片流进行推理（使用numpy数组或PIL图像）
        """
        # 确保图像是RGB格式
        if isinstance(image, np.ndarray):
            # OpenCV图像是BGR格式，转换为RGB
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            pil_image = image

        # 创建临时文件，确保质量为最高
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            pil_image.save(tmp.name, quality=100, subsampling=0)
            image_path = tmp.name

        try:
            # 调用predict方法进行推理
            results = self.predict(
                image_path=image_path,
                threshold=threshold,
                save_result=save_result,
                output_dir=output_dir,
                slice_infer=slice_infer
            )
        finally:
            if os.path.exists(image_path):
                os.unlink(image_path)

        return results
    
    def _format_results(self, results: List[Dict], threshold: float) -> List[Dict]:
        """
        格式化推理结果
    
        Args:
            results (List[Dict]): 原始推理结果
            threshold (float): 检测阈值

        Returns:
            List[Dict]: 格式化后的结果
        """
        formatted_results = []
        for result in results:
            if 'bbox' not in result:
                continue

            bbox_array = result['bbox']
        
            # 确保bbox_array是二维数组
            if isinstance(bbox_array, (list, np.ndarray)) and len(bbox_array) > 0:
                # 如果是一维数组，转换为二维
                if not isinstance(bbox_array[0], (list, np.ndarray)):
                    bbox_array = [bbox_array]

                for det in bbox_array:
                    try:
                        class_id = int(det[0])
                        score = float(det[1])
                        box = det[2:6].tolist() if hasattr(det, 'tolist') else det[2:6]

                        if score < threshold:
                            continue
                            
                        box = [int(coord) for coord in box]
                        formatted_results.append({
                            'bbox': box,
                            'score': score,
                            'category_id': class_id,
                            'category': self._get_category_name(class_id)
                        })
                    except (IndexError, TypeError, ValueError) as e:
                        print(f"Error processing detection: {e}, detection: {det}")
                        continue
                        
        return formatted_results
    
    def _get_category_name(self, class_id: int) -> str:
        """获取类别名称"""
        class_names = {
            0: 'ore carrier', 
            1: 'bulk cargo carrier', 
            2: 'general cargo ship', 
            3: 'container ship', 
            4: 'fishing boat', 
            5: 'passenger ship',
        }
        return class_names.get(class_id, f'unknown_{class_id}')
    
    def batch_predict(
        self,
        image_paths: List[str],
        threshold: Optional[float] = None,
        save_result: bool = False,
        output_dir: Optional[str] = None,
        slice_infer: bool = False,
    ) -> List[List[Dict]]:
        """
        批量推理多张图片（使用文件路径）
        
        Args:
            image_paths (List[str]): 图片路径列表
            threshold (float, optional): 检测阈值
            save_result (bool): 是否保存可视化结果
            output_dir (str, optional): 结果保存目录
            
        Returns:
            List[List[Dict]]: 每张图片的检测结果列表
        """
        return [self.predict(img, threshold, save_result, output_dir, slice_infer) for img in image_paths]
    
    def batch_predict_images(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        threshold: Optional[float] = None,
        save_result: bool = False,
        output_dir: Optional[str] = None,
        slice_infer: bool = False,
    ) -> List[List[Dict]]:
        """
        批量推理多张图片（使用numpy数组或PIL图像）
        
        Args:
            images (List): 图片列表，可以是numpy数组或PIL图像
            threshold (float, optional): 检测阈值
            save_result (bool): 是否保存可视化结果
            output_dir (str, optional): 结果保存目录
            
        Returns:
            List[List[Dict]]: 每张图片的检测结果列表
        """
        return [self.predict_image(img, threshold, save_result, output_dir, slice_infer) for img in images]