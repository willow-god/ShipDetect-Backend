import warnings
from typing import List, Dict, Optional
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
            config (dict, optional): 配置字典。如果为None，则使用内置默认配置
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
        # print("Config:", self.cfg)
        # print("Config type:", type(self.cfg))
        self.trainer = Trainer(self.cfg, mode='test')
        self.trainer.load_weights(self.cfg.weights)
        
    # def _get_default_config(self, config_path) -> dict:
    #     """返回默认配置字典"""
    #     with open(config_path, 'r') as f:
    #         config = yaml.safe_load(f)
    #         print("Default config loaded from:", config_path)
    #         print("config type:", type(config))
    #     # 返回字典类型的数据
    #     return config
    
    def _init_config(self, config_path: str, weights_path: str) -> AttrDict:
        """初始化配置"""        
        # 转换为AttrDict
        cfg = load_config(config_path)
        
        # 设置权重路径
        cfg.weights = weights_path
        
        # 设置设备
        cfg.use_gpu = self.use_gpu
        if self.use_gpu:
            paddle.set_device(f'gpu:{self.device_id}')
        else:
            paddle.set_device('cpu')
        
        # 设置默认推理参数
        cfg.draw_threshold = self.threshold
        cfg.infer_dir = None
        cfg.infer_img = None
        cfg.output_dir = 'output'
        cfg.visualize = False
        
        # 检查配置
        check_config(cfg)
        check_gpu(cfg.use_gpu)
        check_version()
        
        return cfg
    
    def predict(
        self,
        image_path: str,
        threshold: Optional[float] = 0.5,
        save_result: bool = False,
        output_dir: Optional[str] = './output/YOLOv8', 
        slice_infer: bool = False,
    ) -> List[Dict]:
        """
        对单张图片进行推理
        
        Args:
            image_path (str): 图片路径
            threshold (float, optional): 检测阈值，如果为None则使用初始化时的阈值
            save_result (bool): 是否保存可视化结果
            output_dir (str, optional): 结果保存目录
            
        Returns:
            List[Dict]: 检测结果列表，每个元素包含:
                - 'bbox': [x1, y1, x2, y2] 坐标
                - 'score': 置信度
                - 'category_id': 类别ID
                - 'category': 类别名称
        """
        # 设置阈值
        draw_threshold = threshold if threshold is not None else self.threshold
        
        # 设置输出目录
        if output_dir is not None:
            self.cfg.output_dir = output_dir
        
        # 设置是否保存结果
        self.cfg.visualize = save_result
        
        # 执行推理，选择是否切片推理
        if slice_infer:
            # 切片推理的实现
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
        
        # 格式化结果
        # print("Results:", results)
        formatted_results = []
        for result in results:
            if 'bbox' not in result:
                continue

            # bbox数组的格式是 [N,6]，每行: [class_id, score, x1, y1, x2, y2]
            bbox_array = result['bbox']
            
            for det in bbox_array:
                class_id = int(det[0])
                score = float(det[1])
                box = det[2:6].tolist() # 提取坐标

                if score < threshold:
                    continue
                else:
                    # 将坐标转换为整数
                    box = [int(coord) for coord in box]
            
                formatted_results.append({
                    'bbox': box,
                    'score': score,
                    'category_id': class_id,
                    'category': self._get_category_name(class_id)
                })
        
        return formatted_results
    
    def _get_category_name(self, class_id: int) -> str:
        """获取类别名称"""
        # 根据你的数据集修改这个映射
        class_names = {
            0: 'ore carrier', 
            1: 'bulk cargo carrier', 
            2: 'general cargo ship', 
            3: 'container ship', 
            4: 'fishing boat', 
            5: 'passenger ship',
            # 添加其他类别...
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
        批量推理多张图片
        
        Args:
            image_paths (List[str]): 图片路径列表
            threshold (float, optional): 检测阈值
            save_result (bool): 是否保存可视化结果
            output_dir (str, optional): 结果保存目录
            
        Returns:
            List[List[Dict]]: 每张图片的检测结果列表
        """
        return [self.predict(img, threshold, save_result, output_dir, slice_infer) for img in image_paths]