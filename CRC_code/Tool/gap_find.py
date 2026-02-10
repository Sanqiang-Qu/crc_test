import numpy as np

def expand_gap_regions(indices, array_length, extra_gap=0):
    """
    扩展连续的索引区域
    
    参数:
    indices: 包含gap的索引列表（一维数组）
    array_length: 数组的总长度（行数或列数）
    extra_gap: 正整数，用于扩展gap区域的范围
    
    返回:
    expanded_regions: 扩展后的区域列表，每个区域为字典，包含:
        - 'original_start': 原始起始索引
        - 'original_end': 原始结束索引
        - 'original_length': 原始长度
        - 'expanded_start': 扩展后起始索引
        - 'expanded_end': 扩展后结束索引
        - 'expanded_length': 扩展后长度
    """
    expanded_regions = {}
    
    if len(indices) == 0:
        return expanded_regions
    
    # 将连续的索引分组
    groups = []
    current_group = [indices[0]]
    
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current_group.append(indices[i])
        else:
            groups.append(current_group)
            current_group = [indices[i]]
    
    groups.append(current_group)
    
    # 处理每个连续区域
    for idx, group in enumerate(groups):
        if len(group) > 0:
            start = group[0]
            end = group[-1]
            length = len(group)
            
            # 扩展并确保不超出边界
            new_start = max(0, start - extra_gap)
            new_end = min(array_length - 1, end + extra_gap)
            expanded_length = new_end - new_start + 1
            
            expanded_regions[idx] = {
                'original_start': start,
                'original_end': end,
                'original_length': length,
                'expanded_start': new_start,
                'expanded_end': new_end,
                'expanded_length': expanded_length
            }
    
    return expanded_regions


def apply_gap_mask(updated_pattern, final_mask, regions, is_horizontal=True):
    """
    将gap区域应用到图像和mask上
    
    参数:
    updated_pattern: 要更新的图像数组
    final_mask: 要更新的mask数组
    regions: 扩展后的区域字典
    is_horizontal: 是否为水平方向（行），False为垂直方向（列）
    """
    for idx, region_info in regions.items():
        new_start = region_info['expanded_start']
        new_end = region_info['expanded_end']
        
        if is_horizontal:
            # 水平方向：处理行
            final_mask[new_start:new_end+1, :] = 1
            updated_pattern[new_start:new_end+1, :] = -1
        else:
            # 垂直方向：处理列
            final_mask[:, new_start:new_end+1] = 1
            updated_pattern[:, new_start:new_end+1] = -1


def process_eiger2_gap(pattern, extra_gap=3):
    """
    处理EIGER2探测器的gap区域
    
    参数:
    pattern: 二维numpy数组，包含衍射强度数据
    extra_gap: 正整数，用于扩展gap区域的范围
    
    返回:
    result_dict: 包含所有处理结果的字典，包含以下键:
        - 'horizontal_gaps': 横向gap区域的字典
        - 'vertical_gaps': 纵向gap区域的字典
        - 'extra_gap': 输入的扩展值
        - 'updated_pattern': 更新后的pattern数组（gap区域设置为-1）
        - 'mask_matrix': 最终的mask矩阵（gap区域为1，有效区域为0，符合PyFAI要求）
        - 'mask_bool': 布尔mask矩阵（True=gap区域，False=有效区域）
        - 'pattern_shape': 原始图像的形状
        - 'gap_value': 使用的gap阈值
        - 'gap_pixel_count': gap像素总数
        - 'valid_pixel_count': 有效像素总数
        - 'gap_percentage': gap像素占总像素的百分比
    """
    
    pattern = pattern.astype(np.float64)
    pattern[pattern<0] = 2**32 - 1
    
    # 1. 获取gap区的全部索引
    gap_value = 2**32 - 2 
    
    # 获取数组形状
    n_rows, n_cols = pattern.shape
    
    # 创建pattern的副本，避免修改原始数据
    updated_pattern = pattern.copy()
    
    # 初始化最终的mask矩阵（全0，表示都是有效像素，符合PyFAI要求：0=有效，1=屏蔽）
    final_mask = np.zeros_like(pattern, dtype=np.int32)
    
    # 2. 按行统计横向gap区（矩阵化计算）
    horizontal_gaps = {}
    
    # 计算每行的平均值
    row_means = np.mean(pattern, axis=1)
    
    # 找出平均值大于gap_value的行（这些行包含gap）
    gap_rows = row_means > gap_value    
    
    # 找到包含gap的行的索引
    gap_row_indices = np.where(gap_rows)[0]    
    
    # 3. 按列统计纵向gap区（矩阵化计算）
    vertical_gaps = {}
    
    # 计算每列的平均值
    col_means = np.mean(pattern, axis=0)
    
    # 找出平均值大于gap_value的列（这些列包含gap）
    gap_cols = col_means > gap_value
    
    # 找到包含gap的列的索引
    gap_col_indices = np.where(gap_cols)[0]    
    
    # 使用函数扩展横向gap区域
    horizontal_gaps = expand_gap_regions(gap_row_indices, n_rows, extra_gap)
    
    # 应用横向gap mask
    apply_gap_mask(updated_pattern, final_mask, horizontal_gaps, is_horizontal=True)
    

    # 使用函数扩展纵向gap区域
    vertical_gaps = expand_gap_regions(gap_col_indices, n_cols, extra_gap)
    
    # 应用纵向gap mask
    apply_gap_mask(updated_pattern, final_mask, vertical_gaps, is_horizontal=False)   
    
    
    # 创建布尔mask矩阵
    bool_mask = final_mask.astype(bool)
    
    
    updated_pattern[updated_pattern>gap_value] = -1
    
    # 构建返回字典
    result_dict = {
        'horizontal_gaps': horizontal_gaps,
        'vertical_gaps': vertical_gaps,
        'extra_gap': extra_gap,
        'updated_pattern': updated_pattern,
        'mask_matrix': final_mask,  # 整数mask: 1表示gap区域（屏蔽），0表示有效区域
        'mask_bool': bool_mask,     # 布尔mask: True表示gap区域，False表示有效区域
        'pattern_shape': pattern.shape,
        'gap_value': gap_value,
        'row_means': row_means,     # 添加行平均值用于调试
        'col_means': col_means      # 添加列平均值用于调试
    }
    
    return result_dict


# %% 使用示例
import os
from tkinter import filedialog
from PIL import Image


if __name__ == "__main__":
    
    file_path = filedialog.askopenfilename(
        title='Select a TIFF file', filetypes=[('TIFF files', '*.tif *.tiff'), ('All files', '*.*')])
    
    if file_path:
        pathName = os.path.dirname(file_path)
        fileName = os.path.basename(file_path)
        baseName = os.path.splitext(fileName)[0]
        
        # 加载图像
        pattern = np.array(Image.open(file_path))
        
        # 使用函数处理，设置extra_gap=2
        result_dict = process_eiger2_gap(pattern, extra_gap=2)
        
        # 打印结果信息
        # print_gap_info(result_dict)
        
        # 保存结果到文件
        # save_gap_results(result_dict, pathName, baseName)
    else:
        print("未选择文件")