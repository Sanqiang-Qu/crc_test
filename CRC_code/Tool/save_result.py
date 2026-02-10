# save_result.py
import os
import json
import numpy as np
from PIL import Image
import pickle

class ResultSaver:
    """结果保存模块 - 简化和优化版本"""
    
    def __init__(self, main_window):
        self.main = main_window
    
    def save_all_results(self):
        """一键保存所有结果"""
        try:
            # 检查必要数据
            if not self._validate_data():
                return False
            
            # 创建分析文件夹
            analysis_folder = self._create_analysis_folder()
            
            # 保存所有内容
            self._save_figures(analysis_folder)
            self._save_data_files(analysis_folder)
            self._save_parameters(analysis_folder)
            
            # 显示成功信息
            self._show_success_message(analysis_folder)
            return True
            
        except Exception as e:
            self._show_error_message(f"保存过程中出错: {str(e)}")
            return False
    
    def _validate_data(self):
        """验证必要数据是否存在"""
        if not hasattr(self.main, 'pattern') or self.main.pattern is None:
            self._show_error_message("请先加载图像数据")
            return False
        
        if not hasattr(self.main, 'fileName'):
            self._show_error_message("文件信息缺失")
            return False
            
        return True
    
    def _create_analysis_folder(self):
        """创建分析文件夹"""
        base_name = os.path.splitext(self.main.fileName)[0]
        analysis_folder = os.path.join(self.main.pathName, f"{base_name}_analysis")
        os.makedirs(analysis_folder, exist_ok=True)
        return analysis_folder
    
    def _save_figures(self, folder_path):
        """保存所有图形"""
        base_name = os.path.splitext(self.main.fileName)[0]
        figures_info = [
            ('fig1', f'{base_name}_pixel_2D_pattern.png'),
            ('fig2', f'{base_name}_qz-qxy_2D_pattern.png'), 
            ('fig3', f'{base_name}_qz-qy_2D_pattern.png'),
            ('fig4', f'{base_name}_I-q_1D_integration.png'),
            ('fig5', f'{base_name}_I-phi_1D_integration.png')
        ]
        
        for fig_attr, filename in figures_info:
            if hasattr(self.main, fig_attr) and getattr(self.main, fig_attr) is not None:
                fig = getattr(self.main, fig_attr)
                fig.savefig(
                    os.path.join(folder_path, filename),
                    dpi=200, 
                    bbox_inches='tight',
                    facecolor='white'
                )
        
        # 保存原始图像
        Image.fromarray(self.main.pattern).save(
            os.path.join(folder_path, f"{base_name}.tif")
        )
    
    def _save_data_files(self, folder_path):
        """保存数据文件"""
        base_name = os.path.splitext(self.main.fileName)[0]
        
        # 保存I-q数据
        if hasattr(self.main, 'Im') and self.main.Im is not None:
            header = f"# I-q data for {base_name}\n# q(A^-1)\tIntensity"
            np.savetxt(
                os.path.join(folder_path, f"{base_name}_I-q_data.txt"),
                self.main.Im, 
                fmt='%.6f', 
                delimiter='\t',
                header=header
            )
        
        # 保存I-phi数据
        if hasattr(self.main, 'In') and self.main.In is not None:
            header = f"# I-phi data for {base_name}\n# Phi(deg)\tIntensity"
            np.savetxt(
                os.path.join(folder_path, f"{base_name}_I-phi_data.txt"),
                self.main.In,
                fmt='%.6f',
                delimiter='\t', 
                header=header
            )
    
    def _save_parameters(self, folder_path):
        """保存参数配置"""
        base_name = os.path.splitext(self.main.fileName)[0]
        parameters = self._collect_essential_parameters()
        
        # 保存为JSON
        json_file = os.path.join(folder_path, f"{base_name}_parameters.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(parameters, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存为pickle
        pickle_file = os.path.join(folder_path, f"{base_name}_parameters.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(parameters, f)
        
        # 保存简明文本版本
        self._save_parameters_summary(folder_path, base_name, parameters)
    
    def _collect_essential_parameters(self):
        """收集核心参数"""
        return {
            'file_info': {
                'name': getattr(self.main, 'fileName', ''),
                'path': getattr(self.main, 'pathName', '')
            },
            'geometry': self._get_geometry_params(),
            'intensity': self._get_intensity_params(),
            'display_ranges': self._get_display_ranges(),
            'integration': self._get_integration_params(),
            'status': self._get_calculation_status(),
            'statistics': self._get_data_statistics()
        }
    
    def _get_geometry_params(self):
        """获取几何参数"""
        return {
            'center_y': self.main.edit_centery.value(),
            'center_z': self.main.edit_centerz.value(),
            'distance': self.main.edit_distance.value(),
            'pixel_size': self.main.edit_pixel.value(),
            'wavelength': self.main.edit_lambda_Xray.value(),
            'incident_angle': self.main.edit_incident_angle.value()
        }
    
    def _get_intensity_params(self):
        """获取强度参数"""
        return {
            'I_min': self.main.edit_I_min.value(),
            'I_max': self.main.edit_I_max.value()
        }
    
    def _get_display_ranges(self):
        """获取显示范围"""
        return {
            'qxy1': {
                'min': self.main.edit_qxy1_min.value(),
                'max': self.main.edit_qxy1_max.value()
            },
            'qz1': {
                'min': self.main.edit_qz1_min.value(),
                'max': self.main.edit_qz1_max.value()
            },
            'qy2': {
                'min': self.main.edit_qy2_min.value(),
                'max': self.main.edit_qy2_max.value()
            },
            'qz2': {
                'min': self.main.edit_qz2_min.value(),
                'max': self.main.edit_qz2_max.value()
            }
        }
    
    def _get_integration_params(self):
        """获取积分参数"""
        return {
            'radial': {
                'angle': self.main.edit_radi_integ_angle.value(),
                'range': self.main.edit_radi_integ_range.value()
            },
            'azimuthal': {
                'q': self.main.edit_azuth_integ_q.value(),
                'range': self.main.edit_azuth_integ_q_range.value()
            }
        }
    
    def _get_calculation_status(self):
        """获取计算状态"""
        return {
            'calibration_done': hasattr(self.main, 'q') and self.main.q is not None,
            'radial_done': hasattr(self.main, 'Im') and self.main.Im is not None,
            'azimuthal_done': hasattr(self.main, 'In') and self.main.In is not None
        }
    
    def _get_data_statistics(self):
        """获取数据统计信息"""
        stats = {}
        if hasattr(self.main, 'pattern') and self.main.pattern is not None:
            stats['pattern'] = {
                'shape': self.main.pattern.shape,
                'min': float(np.nanmin(self.main.pattern)),
                'max': float(np.nanmax(self.main.pattern)),
                'mean': float(np.nanmean(self.main.pattern))
            }
        return stats
    
    def _save_parameters_summary(self, folder_path, base_name, parameters):
        """保存参数摘要"""
        summary_file = os.path.join(folder_path, f"{base_name}_parameters_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("TwoDpattern Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            # 文件信息
            f.write("FILE INFORMATION:\n")
            f.write(f"  Name: {parameters['file_info']['name']}\n")
            f.write(f"  Path: {parameters['file_info']['path']}\n\n")
            
            # 几何参数
            f.write("GEOMETRY PARAMETERS:\n")
            geo = parameters['geometry']
            for key, value in geo.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 计算状态
            f.write("CALCULATION STATUS:\n")
            status = parameters['status']
            for key, value in status.items():
                f.write(f"  {key}: {value}\n")
    
    def _show_success_message(self, folder_path):
        """显示成功消息"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(
            self.main, "保存完成",
            f"所有结果已保存到:\n{folder_path}",
            QMessageBox.Ok
        )
        
        # 激活批处理功能
        if hasattr(self.main, 'btn_analysis_batch'):
            self.main.btn_analysis_batch.setEnabled(True)
    
    def _show_error_message(self, message):
        """显示错误消息"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(self.main, "保存错误", message, QMessageBox.Ok)