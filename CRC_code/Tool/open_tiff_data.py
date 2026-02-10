# open_tiff_data.py
import os
import numpy as np
from PIL import Image
import SAXS_CRC as sc


def load_tiff_data(file_path):
    """
    独立的数据加载函数 - 从TIFF文件读取数据并生成坐标网格
    
    参数:
        file_path: TIFF文件路径
        
    返回:
        dict: 包含以下键的字典
            - 'pattern': 图像数据数组
            - 'Y': Y坐标网格
            - 'Z': Z坐标网格  
            - 'pathName': 文件所在目录
            - 'fileName': 文件名
    """
    try:
        # 读取文件信息
        pathName = os.path.dirname(file_path)
        fileName = os.path.basename(file_path)
        
        # 加载TIFF图像数据
        pattern = np.array(Image.open(file_path))
        
        # 生成坐标网格
        Y, Z = np.meshgrid(
            sc.arange(0, 1, pattern.shape[1]-1), 
            sc.arange(0, 1, pattern.shape[0]-1)
        )
        
        # 异常值处理
        pattern[pattern > 4.2e9] = -1
        
        return {
            'pattern': pattern,
            'Y': Y,
            'Z': Z,
            'pathName': pathName,
            'fileName': fileName
        }
        
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None

def process_loaded_data(data_dict):
    """
    数据处理验证函数
    
    参数:
        data_dict: load_tiff_data返回的数据字典
        
    返回:
        bool: 数据处理是否成功
    """
    if data_dict is None:
        return False
        
    required_keys = ['pattern', 'Y', 'Z', 'pathName', 'fileName']
    if not all(key in data_dict for key in required_keys):
        print("数据字典缺少必要键")
        return False
        
    if data_dict['pattern'] is None or data_dict['pattern'].size == 0:
        print("图像数据为空")
        return False
        
    return True