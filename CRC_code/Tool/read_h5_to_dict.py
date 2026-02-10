# read_h5_to_dict.py
import fabio
import numpy as np
from tkinter import filedialog
import os


def get_all_files_in_folder(folder_path=None, filter_str=None):
    """
    获取文件夹中所有子目录下的所有文件信息，并可筛选包含特定字符串的文件
    
    参数:
        folder_path: 文件夹路径，如果为None则弹出文件夹选择对话框
        filter_str: 筛选字符串，只返回包含此字符串的文件。如果为None，则返回所有文件
        
    返回:
        包含三个键的字典:
        - 'total_path': 打开的文件夹的完整路径
        - 'file_list': 所有文件去掉总路径后的相对路径列表
        - 'filtered_h5_files': 筛选后包含指定字符串的文件列表
    """
    # 如果未提供文件夹路径，弹出文件夹选择对话框
    if folder_path is None:
        folder_path = filedialog.askdirectory(
            title='选择文件夹'
        )
        if not folder_path:
            print("没有选择文件夹")
            return {'total_path': '', 'file_list': [], 'filtered_h5_files': []}
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return {'total_path': folder_path, 'file_list': [], 'filtered_h5_files': []}
    
    # 规范化路径，确保格式一致
    folder_path = os.path.normpath(folder_path)
    
    # 获取所有文件的路径
    all_files = []
    filtered_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            
            # 计算相对于总路径的相对路径
            rel_path = os.path.relpath(file_path, folder_path)
            
            # 标准化路径分隔符，确保在Windows上使用反斜杠
            rel_path = rel_path.replace('\\', '/')
            
            all_files.append(rel_path)
            
            # 如果提供了筛选字符串，检查文件名是否包含该字符串
            if filter_str and filter_str in file:
                filtered_files.append(rel_path)
    
    # 按字母顺序排序
    all_files.sort()
    filtered_files.sort()
    
    # 创建返回的字典
    result = {
        'total_path': folder_path,
        'file_list': all_files,
        'filtered_h5_files': filtered_files
    }    
    return result



def read_h5_to_dict(file_path=None):
    """
    读取HDF5文件并将三维矩阵转换为字典
    
    参数:
        file_path: 文件路径，如果为None则弹出文件选择对话框
        
    返回:
        包含多个二维矩阵的字典，键名为'data_0', 'data_1', ...
    
    示例:
        如果输入是(3, 2162, 2068)的三维矩阵，返回:
        {
            'data_0': shape(2162, 2068),
            'data_1': shape(2162, 2068), 
            'data_2': shape(2162, 2068)
        }
    """
    # 如果未提供文件路径，弹出文件选择对话框
    if file_path is None:
        file_path = filedialog.askopenfilename(
            title='Select a HDF5 file', 
            filetypes=[('HDF5 files', '*.hdf5 *.h5 *.hdf'), 
                       ('All files', '*.*')]
        )
        if not file_path:
            return None
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        # 使用fabio打开Eiger探测器数据
        fabio_obj = fabio.open(file_path)
        data = fabio_obj.data
        
        # 获取数据的形状
        original_shape = data.shape
        print(f"原始数据形状: {original_shape}")
        print(f"数据类型: {data.dtype}")
        
        # 处理不同维度的数据
        if data.ndim == 2:
            # 如果是二维数据，直接放入字典
            data_dict = {'data_0': data}
            
        elif data.ndim == 3:
            # 如果是三维数据，拆分为多个二维矩阵
            n_frames = data.shape[0]
            data_dict = {}
            
            for i in range(n_frames):
                key = f'data_{i}'
                data_dict[key] = data[i]
            
            print(f"三维数据，拆分为{n_frames}个二维矩阵")
            
        else:
            # 处理其他维度的情况
            print(f"警告: 数据维度为{data.ndim}，尝试处理为三维数据...")
            
            # 如果第一维是1，尝试压缩掉
            if data.shape[0] == 1 and data.ndim >= 3:
                data = np.squeeze(data, axis=0)
                print(f"压缩后形状: {data.shape}")
                
                # 递归调用处理压缩后的数据
                return read_h5_to_dict(file_path)
            else:
                # 其他情况，将前两维之外的所有维度展平
                if data.ndim > 3:
                    # 计算新的形状
                    n_frames = np.prod(data.shape[:-2])
                    new_shape = (n_frames, data.shape[-2], data.shape[-1])
                    data = data.reshape(new_shape)
                    print(f"重塑后形状: {data.shape}")
                    
                    # 递归调用处理重塑后的数据
                    return read_h5_to_dict(file_path)
                else:
                    raise ValueError(f"不支持的维度: {data.ndim}")
        
        # 打印统计信息
        print("\n字典内容:")
        for key, value in data_dict.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                  f"min={value.min():.2f}, max={value.max():.2f}, mean={value.mean():.2f}")
        
        return data_dict
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_dict_to_tif(data_dict, channel=0, input_file_path=None):
    """
    将字典中指定通道的二维矩阵保存为TIFF格式，不修改任何数值
    
    参数:
        data_dict: 包含二维矩阵的字典
        channel: 要保存的通道索引（0对应第一个字典，1对应第二个字典，以此类推）
        input_file_path: 输入文件路径，用于生成输出路径。如果为None，则让用户选择保存位置
    """
    if not data_dict:
        print("字典为空，无法保存")
        return
    
    # 检查通道索引是否有效
    channel_key = f'data_{channel}'
    if channel_key not in data_dict:
        print(f"错误: 字典中没有通道 {channel} (键: {channel_key})")
        print(f"可用的通道: {list(data_dict.keys())}")
        return
    
    # 获取要保存的二维矩阵
    matrix_to_save = data_dict[channel_key]
    print(f"准备保存通道 {channel}: 形状={matrix_to_save.shape}, 类型={matrix_to_save.dtype}")
    
    # 确定输出路径
    if input_file_path is not None and os.path.exists(input_file_path):
        # 基于输入文件路径生成输出路径
        dir_path = os.path.dirname(input_file_path)
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_path = os.path.join(dir_path, f"{base_name}_channel_{channel}.tif")
    else:
        # 如果没有输入文件路径，弹出保存对话框
        output_path = filedialog.asksaveasfilename(
            title=f'保存通道 {channel} 为TIFF文件',
            defaultextension=".tif",
            filetypes=[('TIFF files', '*.tif *.tiff'), ('All files', '*.*')],
            initialfile=f"channel_{channel}.tif"
        )
        if not output_path:
            print("用户取消了保存")
            return
    
    try:
        # 尝试使用tifffile库保存为TIFF
        try:
            import tifffile
            # 使用tifffile直接保存，不修改任何数值
            tifffile.imwrite(output_path, matrix_to_save)
            print(f"使用tifffile保存成功: {output_path}")
            print(f"文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        except ImportError:
            # 如果tifffile不可用，尝试使用PIL
            print("tifffile库未安装，尝试使用PIL...")
            print("注意: PIL不支持某些数据类型(如uint32)，可能会自动转换数据类型")
            print("建议安装tifffile库以保存原始数据: pip install tifffile")
            
            try:
                from PIL import Image
                # PIL不支持的数据类型会报错，但我们不进行任何转换
                # 让PIL自己处理，如果失败就报错
                img = Image.fromarray(matrix_to_save)
                img.save(output_path)
                print(f"使用PIL保存成功: {output_path}")
                print(f"注意: PIL可能已自动转换了数据类型")
                print(f"原始数据类型: {matrix_to_save.dtype}")
                print(f"保存后数据类型: 请检查文件确认")
                
            except Exception as pil_error:
                print(f"PIL保存失败: {pil_error}")
                print(f"原始数据类型 {matrix_to_save.dtype} 可能不被PIL支持")
                print("建议安装tifffile库: pip install tifffile")
                return
        
    except Exception as e:
        print(f"保存文件时出错: {e}")
        import traceback
        traceback.print_exc()


def batch_convert_h5_to_tif(files_info_dict, channel_selected=0):
    """
    批量将HDF5文件转换为TIFF格式，只保存指定通道
    
    参数:
        files_info_dict: get_all_files_in_folder函数返回的字典
        channel_selected: 要保存的通道索引（默认0，即第一个通道）
        
    返回:
        转换结果统计字典
    """
    if not files_info_dict or 'total_path' not in files_info_dict:
        print("无效的文件信息字典")
        return {'success': 0, 'failed': 0, 'total': 0}
    
    total_path = files_info_dict['total_path']
    filtered_files = files_info_dict.get('filtered_h5_files', [])
    
    if not filtered_files:
        print("没有找到需要转换的HDF5文件")
        return {'success': 0, 'failed': 0, 'total': 0}
    
    # 创建输出目录
    output_dir = os.path.join(total_path, "exacted_batch_h5")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始批量转换HDF5文件到TIFF格式")
    print(f"总路径: {total_path}")
    print(f"输出目录: {output_dir}")
    print(f"待转换文件数量: {len(filtered_files)}")
    print(f"保存通道: {channel_selected}")
    
    # 统计结果
    results = {
        'success': 0,
        'failed': 0,
        'total': len(filtered_files),
        'output_dir': output_dir,
        'channel_selected': channel_selected
    }
    
    # 处理每个文件
    for i, rel_path in enumerate(filtered_files):
        # 构建完整文件路径
        full_path = os.path.join(total_path, rel_path)
        
        print(f"\n[{i+1}/{len(filtered_files)}] 处理文件: {rel_path}")
        
        try:
            # 读取HDF5文件
            data_dict = read_h5_to_dict(full_path)
            
            if data_dict is None:
                print(f"  读取失败: {rel_path}")
                results['failed'] += 1
                continue
            
            # 检查指定通道是否存在
            channel_key = f'data_{channel_selected}'
            if channel_key not in data_dict:
                print(f"  警告: 文件 {rel_path} 中没有通道 {channel_selected}")
                print(f"  可用通道: {list(data_dict.keys())}")
                results['failed'] += 1
                continue
            
            # 获取文件的基本名称（不含扩展名）
            # base_name = os.path.splitext(os.path.basename(rel_path))[0]
            
            # # 构建输出文件路径
            # output_filename = f"{base_name}_channel_{channel_selected}.tif"
            # output_path = os.path.join(output_dir, output_filename)

            # 解析文件路径，获取文件夹名和文件名
            file_path_parts = rel_path.split('/')
            if len(file_path_parts) > 1:
                # 提取文件夹名（最后一个路径部分）和文件名
                folder_name = file_path_parts[-2]
                file_name = file_path_parts[-1]
                base_name = os.path.splitext(file_name)[0]
                # 构建包含文件夹名的输出文件名
                output_filename = f"{folder_name}_{base_name}_channel_{channel_selected}.tif"
            else:
                # 没有文件夹结构，使用原始文件名
                base_name = os.path.splitext(os.path.basename(rel_path))[0]
                output_filename = f"{base_name}_channel_{channel_selected}.tif"

            # 构建输出文件路径
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存指定通道为TIFF
            try:
                # 使用tifffile库保存
                import tifffile
                tifffile.imwrite(output_path, data_dict[channel_key])
                print(f"  保存成功: {output_filename}")
                results['success'] += 1
            except ImportError:
                # 回退到PIL
                try:
                    from PIL import Image
                    img = Image.fromarray(data_dict[channel_key])
                    img.save(output_path)
                    print(f"  保存成功: {output_filename} (使用PIL)")
                    results['success'] += 1
                except Exception as e:
                    print(f"  保存失败: {e}")
                    results['failed'] += 1
            
        except Exception as e:
            print(f"  处理失败: {rel_path}, 错误: {e}")
            results['failed'] += 1
    
    # 打印统计信息
    print(f"\n批量转换完成!")
    print(f"总文件数: {results['total']}")
    print(f"成功转换: {results['success']} 个文件")
    print(f"失败: {results['failed']}")
    print(f"输出目录: {results['output_dir']}")
    print(f"保存的通道: {results['channel_selected']}")
    
    return results


# %% 主函数，可以直接运行
if __name__ == "__main__":
    # 使用示例1: 读取HDF5文件
    h5_file_index = 'data_000001.h5'
    channel_to_save = 0
    
    
    all_data_file = get_all_files_in_folder(folder_path=None, filter_str=h5_file_index)
    batch_results = batch_convert_h5_to_tif(all_data_file, channel_selected=channel_to_save)
    
    
            
            
            
            
            