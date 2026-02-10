import numpy as np
import Tool.SAXS_CRC_for_twodpattern as sc
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QLabel, QVBoxLayout, QGridLayout, QSplitter, QSizePolicy,
                             QSpacerItem, QMessageBox, QDoubleSpinBox, QDialog)
from PyQt5.QtCore import Qt


def pattern_flip(pattern, num):
    if num == 1:  # 逆时针90°
        # 转置后上下翻转
        return pattern.T[::-1, :]
    elif num == 2:  # 逆时针180°
        # 上下翻转后左右翻转
        return pattern[::-1, ::-1]
    elif num == 3:  # 逆时针270°
        # 转置后左右翻转
        return pattern.T[:, ::-1]
    elif num == 4:  # 上下翻转（左右不动）
        # 仅第一维（行）反转，第二维（列）不变
        return pattern[::-1, :]
    elif num == 5:  # 左右翻转（上下不动）
        # 仅第二维（列）反转，第一维（行）不变
        return pattern[:, ::-1]
    else:
        raise ValueError("num 必须是 1, 2, 3, 4 或 5")



# %% 图像旋转模块
class PatternRotator:
    """图像旋转和PONI参数更新模块"""
    
    def __init__(self, parent_window, pattern):
        self.parent_window = parent_window
        self.pattern = pattern
        self.rotate_pattern = None
        self.Y = None
        self.Z = None
    
    def rotate_pattern_and_update_poni(self, num):
        """执行图像旋转并更新PONI参数"""
        # 调用旋转函数
        self.rotate_pattern = pattern_flip(self.pattern, num)
        self.Y, self.Z = np.meshgrid(
            sc.arange(0, 1, self.rotate_pattern.shape[1]-1), 
            sc.arange(0, 1, self.rotate_pattern.shape[0]-1))
        
        # 如果存在PONI参数，则更新
        if hasattr(self.parent_window, 'poni_para') and self.parent_window.poni_para is not None:
            self.update_poni_parameters(num)
        
        return self.rotate_pattern, self.Y, self.Z
    
    def update_poni_parameters(self, num):
        """根据旋转类型更新PONI参数"""
        try:
            if num == 1:  # 顺时针旋转90度
                self._update_poni_rotate_90()
            elif num == 2:  # 顺时针旋转180度
                self._update_poni_rotate_180()
            elif num == 3:  # 顺时针旋转270度
                self._update_poni_rotate_270()
            elif num == 4:  # 上下翻转
                self._update_poni_flip_vertical()
            elif num == 5:  # 左右翻转
                self._update_poni_flip_horizontal()
            
            # 更新GUI界面参数
            self._update_gui_parameters()
            
        except Exception as e:
            print(f"更新PONI参数时出错: {e}")
            QMessageBox.warning(None, "错误", f"更新PONI参数时出错: {e}", QMessageBox.Ok)
    
    def _update_poni_rotate_90(self):
        """顺时针旋转90度的PONI参数更新"""
        poni_para = self.parent_window.poni_para
        
        # 1. 提取旋转前的图像尺寸
        pattern_height_old = self.pattern.shape[0]  # 原始图像高度
        pattern_width_old = self.pattern.shape[1]   # 原始图像宽度
        
        # 2. 将(poni2_pixel, poni1_pixel)组合成点A
        poni2_pixel_old = poni_para["poni2_pixel"]
        poni1_pixel_old = poni_para["poni1_pixel"]
        point_A = (poni2_pixel_old, poni1_pixel_old)
        
        # 3. 将点A绕原点顺时针旋转90度得到A_rot
        # 顺时针旋转90度： (x, y) -> (y, -x)
        point_A_rot = (point_A[1], -point_A[0])
        
        # 4. A_new = A_rot + (0, shape[1])
        point_A_new = (point_A_rot[0], point_A_rot[1] + pattern_width_old)
        
        # 5. 用A_new[1]更新poni1相关参数，A_new[0]更新poni2相关参数
        poni1_pixel_new = point_A_new[1]  # y坐标
        poni2_pixel_new = point_A_new[0]  # x坐标
        
        # 更新PONI参数
        pixel_size = poni_para["pixel"]
        poni_para["poni1"] = poni1_pixel_new * pixel_size
        poni_para["poni2"] = poni2_pixel_new * pixel_size
        poni_para["poni1_pixel"] = poni1_pixel_new
        poni_para["poni2_pixel"] = poni2_pixel_new
        
        # 同时需要更新center_y和center_z
        poni_para["center_y"] = poni2_pixel_new
        poni_para["center_z"] = poni1_pixel_new
        
        print("顺时针旋转90度 - PONI参数已更新：")
        print(f"  原始点A: ({poni2_pixel_old}, {poni1_pixel_old})")
        print(f"  旋转后A_rot: ({point_A_rot[0]}, {point_A_rot[1]})")
        print(f"  平移后A_new: ({point_A_new[0]}, {point_A_new[1]})")
        print(f"  poni1_pixel: {poni1_pixel_old} -> {poni1_pixel_new}")
        print(f"  poni2_pixel: {poni2_pixel_old} -> {poni2_pixel_new}")
    
    def _update_poni_rotate_180(self):
        """顺时针旋转180度的PONI参数更新"""
        poni_para = self.parent_window.poni_para
        
        # 1. 提取旋转前的图像尺寸
        pattern_height_old = self.pattern.shape[0]  # 原始图像高度
        pattern_width_old = self.pattern.shape[1]   # 原始图像宽度
        
        # 2. 将(poni2_pixel, poni1_pixel)组合成点A
        poni2_pixel_old = poni_para["poni2_pixel"]
        poni1_pixel_old = poni_para["poni1_pixel"]
        point_A = (poni2_pixel_old, poni1_pixel_old)
        
        # 3. 将点A绕原点顺时针旋转180度得到A_rot
        # 顺时针旋转180度： (x, y) -> (-x, -y)
        point_A_rot = (-point_A[0], -point_A[1])
        
        # 4. A_new = A_rot + (shape[1], shape[0])
        point_A_new = (point_A_rot[0] + pattern_width_old, point_A_rot[1] + pattern_height_old)
        
        # 5. 用A_new[1]更新poni1相关参数，A_new[0]更新poni2相关参数
        poni1_pixel_new = point_A_new[1]  # y坐标
        poni2_pixel_new = point_A_new[0]  # x坐标
        
        # 更新PONI参数
        pixel_size = poni_para["pixel"]
        poni_para["poni1"] = poni1_pixel_new * pixel_size
        poni_para["poni2"] = poni2_pixel_new * pixel_size
        poni_para["poni1_pixel"] = poni1_pixel_new
        poni_para["poni2_pixel"] = poni2_pixel_new
        
        # 同时需要更新center_y和center_z
        poni_para["center_y"] = poni2_pixel_new
        poni_para["center_z"] = poni1_pixel_new
        
        print("顺时针旋转180度 - PONI参数已更新：")
        print(f"  原始点A: ({poni2_pixel_old}, {poni1_pixel_old})")
        print(f"  旋转后A_rot: ({point_A_rot[0]}, {point_A_rot[1]})")
        print(f"  平移后A_new: ({point_A_new[0]}, {point_A_new[1]})")
        print(f"  poni1_pixel: {poni1_pixel_old} -> {poni1_pixel_new}")
        print(f"  poni2_pixel: {poni2_pixel_old} -> {poni2_pixel_new}")
    
    def _update_poni_rotate_270(self):
        """顺时针旋转270度的PONI参数更新"""
        poni_para = self.parent_window.poni_para
        
        # 1. 提取旋转前的图像尺寸
        pattern_height_old = self.pattern.shape[0]  # 原始图像高度
        pattern_width_old = self.pattern.shape[1]   # 原始图像宽度
        
        # 2. 将(poni2_pixel, poni1_pixel)组合成点A
        poni2_pixel_old = poni_para["poni2_pixel"]
        poni1_pixel_old = poni_para["poni1_pixel"]
        point_A = (poni2_pixel_old, poni1_pixel_old)
        
        # 3. 将点A绕原点顺时针旋转270度得到A_rot
        # 顺时针旋转270度： (x, y) -> (-y, x)
        point_A_rot = (-point_A[1], point_A[0])
        
        # 4. A_new = A_rot + (shape[0], 0)
        point_A_new = (point_A_rot[0] + pattern_height_old, point_A_rot[1])
        
        # 5. 用A_new[1]更新poni1相关参数，A_new[0]更新poni2相关参数
        poni1_pixel_new = point_A_new[1]  # y坐标
        poni2_pixel_new = point_A_new[0]  # x坐标
        
        # 更新PONI参数
        pixel_size = poni_para["pixel"]
        poni_para["poni1"] = poni1_pixel_new * pixel_size
        poni_para["poni2"] = poni2_pixel_new * pixel_size
        poni_para["poni1_pixel"] = poni1_pixel_new
        poni_para["poni2_pixel"] = poni2_pixel_new
        
        # 同时需要更新center_y和center_z
        poni_para["center_y"] = poni2_pixel_new
        poni_para["center_z"] = poni1_pixel_new
        
        print("顺时针旋转270度 - PONI参数已更新：")
        print(f"  原始点A: ({poni2_pixel_old}, {poni1_pixel_old})")
        print(f"  旋转后A_rot: ({point_A_rot[0]}, {point_A_rot[1]})")
        print(f"  平移后A_new: ({point_A_new[0]}, {point_A_new[1]})")
        print(f"  poni1_pixel: {poni1_pixel_old} -> {poni1_pixel_new}")
        print(f"  poni2_pixel: {poni2_pixel_old} -> {poni2_pixel_new}")
    
    def _update_poni_flip_vertical(self):
        """上下翻转的PONI参数更新"""
        poni_para = self.parent_window.poni_para
        
        # 计算新的poni1_pixel
        pattern_height = self.rotate_pattern.shape[0]
        poni1_pixel_old = poni_para["poni1_pixel"]
        poni1_pixel_new = abs(poni1_pixel_old - pattern_height)
        
        # 更新PONI参数
        pixel_size = poni_para["pixel"]
        poni_para["poni1_pixel"] = poni1_pixel_new
        poni_para["poni1"] = poni1_pixel_new * pixel_size
        
        # poni2保持不变
        # poni_para["poni2"] 不变
        # poni_para["poni2_pixel"] 不变
        
        # 重新计算center_z
        center_z_new = pattern_height - poni_para["center_z"]
        poni_para["center_z"] = center_z_new
        
        print("上下翻转 - PONI参数已更新：")
        print(f"  poni1_pixel: {poni1_pixel_old} -> {poni1_pixel_new}")
        print(f"  poni1: {poni_para['poni1']} mm")
        print(f"  center_z: {poni_para['center_z']}")
    
    def _update_poni_flip_horizontal(self):
        """左右翻转的PONI参数更新"""
        poni_para = self.parent_window.poni_para
        
        # 计算新的poni2_pixel
        pattern_width = self.rotate_pattern.shape[1]
        poni2_pixel_old = poni_para["poni2_pixel"]
        poni2_pixel_new = abs(poni2_pixel_old - pattern_width)
        
        # 更新PONI参数
        pixel_size = poni_para["pixel"]
        poni_para["poni2_pixel"] = poni2_pixel_new
        poni_para["poni2"] = poni2_pixel_new * pixel_size
        
        # poni1保持不变
        # poni_para["poni1"] 不变
        # poni_para["poni1_pixel"] 不变
        
        # 重新计算center_y
        center_y_new = pattern_width - poni_para["center_y"]
        poni_para["center_y"] = center_y_new
        
        print("左右翻转 - PONI参数已更新：")
        print(f"  poni2_pixel: {poni2_pixel_old} -> {poni2_pixel_new}")
        print(f"  poni2: {poni_para['poni2']} mm")
        print(f"  center_y: {poni_para['center_y']}")
    
    def _update_gui_parameters(self):
        """更新GUI界面参数"""
        poni_para = self.parent_window.poni_para
        self.parent_window.edit_centery.setValue(poni_para["center_y"])
        self.parent_window.edit_centerz.setValue(poni_para["center_z"])
        self.parent_window.edit_lambda_Xray.setValue(poni_para["wavelength"])
        self.parent_window.edit_distance.setValue(poni_para["distance"])
        self.parent_window.edit_pixel.setValue(poni_para["pixel"])


# %% 功能二：图像翻转的GUI界面（更新版）
class FlipDialog(QDialog):
    """图像翻转对话框"""
    def __init__(self, parent=None, pattern=None):
        super().__init__(parent)
        self.pattern = pattern
        self.parent_window = parent  # 保存主窗口引用
        self.rotator = PatternRotator(parent, pattern)  # 创建旋转器实例
        self.init_ui()
        self.flip_status = "未翻转"
        self.flip_dialog = None
        
    def init_ui(self):
        self.setWindowTitle("图像翻转")
        self.setFixedSize(300, 200)
        
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("选择旋转角度:")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 旋转按钮
        self.btn_90 = QPushButton("顺时针旋转90度")
        self.btn_180 = QPushButton("顺时针旋转180度")
        self.btn_270 = QPushButton("顺时针旋转270度")
        self.btn_up_dwon = QPushButton("图片上下翻转")
        self.btn_left_right = QPushButton("图片左右翻转")
        
        # 设置按钮样式
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px;
                text-align: center;
                text-decoration: none;
                font-size: 12px;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        
        for btn in [self.btn_90, self.btn_180, self.btn_270, self.btn_up_dwon, self.btn_left_right]:
            btn.setStyleSheet(button_style)
            btn.setFixedHeight(40)
            layout.addWidget(btn)
        
        # 连接按钮信号
        self.btn_90.clicked.connect(lambda: self.rotate_and_close(1))
        self.btn_180.clicked.connect(lambda: self.rotate_and_close(2))
        self.btn_270.clicked.connect(lambda: self.rotate_and_close(3))    
        self.btn_up_dwon.clicked.connect(lambda: self.rotate_and_close(4))
        self.btn_left_right.clicked.connect(lambda: self.rotate_and_close(5))
        
        self.setLayout(layout)
    
    def rotate_and_close(self, num):
        """执行图像旋转"""
        if self.pattern is not None:
            # 使用旋转器执行旋转和PONI参数更新
            self.rotate_pattern, self.Y, self.Z = self.rotator.rotate_pattern_and_update_poni(num)
            # 关闭窗口
            self.accept()            
        else:
            QMessageBox.warning(self, "错误", "没有可用的图像数据", QMessageBox.Ok)
            self.reject()
    
    def get_rotated_data(self):
        """获取旋转后的数据"""
        return self.rotate_pattern, self.Y, self.Z