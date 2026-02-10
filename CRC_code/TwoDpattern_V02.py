# %%导入库
import sys
import os
import json
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, QComboBox, QApplication,
                             QLabel, QVBoxLayout, QGridLayout, QSplitter, QSizePolicy,
                             QSpacerItem, QMessageBox, QDoubleSpinBox, QDialog, QProgressDialog, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from Tool.gap_find import process_eiger2_gap
from Tool.save_result import ResultSaver
from Tool.Calibration import convert_poni_to_fit2d, calibrate_parameters
from Tool.Pic_flip import FlipDialog
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator


# print

extra_gap_value = 5

# %% 后台批处理
class BatchProcessThread(QThread):
    """批处理后台线程"""
    # 信号定义
    progress_updated = pyqtSignal(int, str)  # 进度更新信号 (当前进度, 状态文本)
    process_finished = pyqtSignal(int, int, bool)  # 处理完成信号 (成功数, 总数, 是否取消)
    process_error = pyqtSignal(str)  # 错误信号 (错误信息)
    
    def __init__(self, pathName, poni_file_path, tiff_files, batch_parameters):
        super().__init__()
        self.pathName = pathName
        self.poni_file_path = poni_file_path
        self.tiff_files = tiff_files
        self.batch_parameters = batch_parameters
        self.is_canceled = False
        
    def run(self):
        """线程运行方法"""
        success_count = 0
        total_count = len(self.tiff_files)
        
        try:
            for i, filename in enumerate(self.tiff_files):
                # 检查是否取消
                if self.is_canceled:
                    break
                    
                # 发送进度更新信号
                status_text = f"处理中: {filename} ({i+1}/{total_count})"
                self.progress_updated.emit(i, status_text)
                
                # 执行批处理
                try:
                    result = TwoDpattern_batch_son(
                        self.pathName, 
                        filename, 
                        self.poni_file_path,
                        self.batch_parameters,
                    )
                    
                    if result:
                        success_count += 1
                        print(f"处理成功: {filename} ({i+1}/{total_count})")
                    else:
                        print(f"处理失败: {filename}")
                        
                except Exception as e:
                    print(f"处理错误 {filename}: {str(e)}")
            
            # 发送完成信号
            self.process_finished.emit(success_count, total_count, self.is_canceled)
            
        except Exception as e:
            # 发送错误信号
            self.process_error.emit(str(e))
    
    def cancel(self):
        """取消处理"""
        self.is_canceled = True



# %% 14. 加载默认参数小函数
def load_default_parameters(default_para_file, parameter_mapping):
    """加载默认参数"""
    try:
        if os.path.exists(default_para_file):
            with open(default_para_file, 'r', encoding='utf-8') as f:
                default_parameters = json.load(f)
            
            for param_name, value in default_parameters.items():
                if param_name in parameter_mapping:
                    widget = parameter_mapping[param_name]
                    if isinstance(widget, QComboBox):
                        # 对于QComboBox，设置当前文本
                        index = widget.findText(str(value))
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    else:
                        # 对于QDoubleSpinBox，设置值
                        widget.setValue(value)
            
            default_parameters_loaded = True
            print(f"默认参数已从 {default_para_file} 加载")
        else:
            default_parameters_loaded = False
            print("未找到默认参数文件，将使用硬编码默认值")
    
    except Exception as e:
        default_parameters_loaded = False
        print(f"加载默认参数时出错: {e}，将使用硬编码默认值")
    
    return default_parameters_loaded

# 15. 保存默认参数小函数
def save_default_parameters(default_para_file, parameter_mapping):
    """保存当前参数为默认参数"""
    try:
        default_parameters = {}
        for param_name, widget in parameter_mapping.items():
            if isinstance(widget, QComboBox):
                default_parameters[param_name] = widget.currentText()
            else:
                default_parameters[param_name] = widget.value()
        
        with open(default_para_file, 'w', encoding='utf-8') as f:
            json.dump(default_parameters, f, indent=4, ensure_ascii=False)
        
        print(f"默认参数已保存到: {default_para_file}")
    
    except Exception as e:
        print(f"保存默认参数时出错: {e}")


# %% 主界面设计
class TwoDpattern(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TwoDpattern')
        self.setGeometry(50, 200, 1600, 800)  # 初始窗口尺寸
        
        # 主窗口部件
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        
        # % 上方按钮网格
        button_upper_grid = QGridLayout()
        button_upper_grid.setHorizontalSpacing(30)
        button_upper_grid.setVerticalSpacing(10)
        
        self.btn_load = QPushButton("导入数据")        
        self.btn_read_poni = QPushButton("读取PONI文件")
        self.btn_calibrate = QPushButton("计算散射矢量与保存配置")
        self.btn_wait_2 = QPushButton("待开发2")
        self.btn_save = QPushButton("单次结果保存")
        self.btn_analysis_batch = QPushButton("批处理")        
        
        button_upper_grid.addWidget(self.btn_load, 0, 0)        
        button_upper_grid.addWidget(self.btn_read_poni, 0, 1)
        button_upper_grid.addWidget(self.btn_calibrate, 0, 2)
        button_upper_grid.addWidget(self.btn_wait_2, 0, 3)
        button_upper_grid.addWidget(self.btn_save, 0, 4)
        button_upper_grid.addWidget(self.btn_analysis_batch, 0, 5)        
        
        # % 下方绘图和参数区
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # 绘图区
        self.fig1 = Figure(figsize=(5, 4)) 
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111)
        # self.fig1, self.ax1 = plt.subplots(figsize=(10, 8))
        # self.fig1.canvas.manager.set_window_title('plot module')
        
        # 创建参数容器
        para_container = QWidget()
        para_layout = QVBoxLayout(para_container)
        para_layout.setContentsMargins(5, 5, 5, 5)  # 统一内边距
        
        # 注释标签-完全居中
        label_note = QLabel("规定X方向为X-ray,Y方向为探测器水平,Z方向为探测器竖直")
        label_note.setAlignment(Qt.AlignCenter)
        label_note.setStyleSheet("font-weight: bold; margin-bottom: 10px;")  # 添加视觉分隔
        para_layout.addWidget(label_note)
        
        # 使用QGridLayout实现3行2列布局
        grid_layout = QGridLayout()
        grid_layout.setVerticalSpacing(2)
        grid_layout.setHorizontalSpacing(5)
        
        # 创建参数控件
        self.edit_centery = QDoubleSpinBox(); self.edit_centerz = QDoubleSpinBox();
        self.edit_distance = QDoubleSpinBox(); self.edit_pixel = QDoubleSpinBox();
        self.edit_lambda_Xray = QDoubleSpinBox(); self.edit_incident_angle = QDoubleSpinBox();
        self.edit_I_min = QDoubleSpinBox(); self.edit_I_max = QDoubleSpinBox(); 
        self.combo_colormap = QComboBox(); self.combo_display_mode = QComboBox();
        self.edit_qxy1_min = QDoubleSpinBox(); self.edit_qxy1_max = QDoubleSpinBox();
        self.edit_qz1_min = QDoubleSpinBox(); self.edit_qz1_max = QDoubleSpinBox();
        self.edit_qy2_min = QDoubleSpinBox(); self.edit_qy2_max = QDoubleSpinBox();
        self.edit_qz2_min = QDoubleSpinBox(); self.edit_qz2_max = QDoubleSpinBox();
        self.edit_radi_integ_angle = QDoubleSpinBox(); self.edit_radi_integ_range = QDoubleSpinBox();
        self.edit_azuth_integ_q = QDoubleSpinBox(); self.edit_azuth_integ_q_range = QDoubleSpinBox();  
        
        for spin_box in [
            self.edit_centery, self.edit_centerz, self.edit_distance, self.edit_pixel,
            self.edit_lambda_Xray, self.edit_incident_angle, self.edit_I_min, self.edit_I_max,
            self.edit_qxy1_min, self.edit_qxy1_max, self.edit_qz1_min, self.edit_qz1_max,
            self.edit_qy2_min, self.edit_qy2_max, self.edit_qz2_min, self.edit_qz2_max,
            self.edit_radi_integ_angle, self.edit_radi_integ_range, self.edit_azuth_integ_q, 
            self.edit_azuth_integ_q_range]:
            spin_box.setRange(-1e10, 1e10); spin_box.setMaximumWidth(100); spin_box.setDecimals(4);
        
        # 1. 径向积分角度、径向积分范围 - 淡红色
        self.edit_radi_integ_angle.setStyleSheet("background-color: #ffcccc;")
        self.edit_radi_integ_range.setStyleSheet("background-color: #ffcccc;")
        
        # 2. 方位积分qc、方位角范围 - 淡蓝色
        self.edit_azuth_integ_q.setStyleSheet("background-color: #cce5ff;")
        self.edit_azuth_integ_q_range.setStyleSheet("background-color: #cce5ff;")
        
        # 3. distance、pixel - 淡黄色
        self.edit_distance.setStyleSheet("background-color: #ffffcc;")
        self.edit_pixel.setStyleSheet("background-color: #ffffcc;")
        
        # 4. 强度下限、强度上限 - 淡绿色
        self.edit_I_min.setStyleSheet("background-color: #ccffcc;")
        self.edit_I_max.setStyleSheet("background-color: #ccffcc;")
        
        # 5. X-ray波长、掠入射角 - 淡紫色
        self.edit_lambda_Xray.setStyleSheet("background-color: #e6ccff;")
        self.edit_incident_angle.setStyleSheet("background-color: #e6ccff;")
        
        # 6. 选择显示模式
        self.combo_display_mode.addItems(["线性", "对数"])
        self.combo_display_mode.setCurrentIndex(0)  # 默认线性模式
        self.combo_display_mode.setStyleSheet("background-color: #e6f7ff;")
        
        # 7. colormap
        self.combo_colormap.setStyleSheet("background-color: #e6f7ff;")
        
        # 6. 其他10个数值框 - 灰色
        gray_boxes = [
            self.edit_centery, self.edit_centerz,
            self.edit_qxy1_min, self.edit_qxy1_max,
            self.edit_qz1_min, self.edit_qz1_max,
            self.edit_qy2_min, self.edit_qy2_max,
            self.edit_qz2_min, self.edit_qz2_max]
        
        for box in gray_boxes:
            box.setStyleSheet("background-color: #e0e0e0;")
        
        grid_layout.addWidget(QLabel("Center_Y:"), 0, 0)
        grid_layout.addWidget(self.edit_centery, 0, 1)        
        grid_layout.addWidget(QLabel("Center_Z:"), 0, 2)
        grid_layout.addWidget(self.edit_centerz, 0, 3)        
        grid_layout.addWidget(QLabel("distance:"), 1, 0)
        grid_layout.addWidget(self.edit_distance, 1, 1)        
        grid_layout.addWidget(QLabel("pixel:"), 1, 2)
        grid_layout.addWidget(self.edit_pixel, 1, 3)        
        grid_layout.addWidget(QLabel("X-ray波长:"), 2, 0)
        grid_layout.addWidget(self.edit_lambda_Xray, 2, 1)        
        grid_layout.addWidget(QLabel("掠入射角:"), 2, 2)
        grid_layout.addWidget(self.edit_incident_angle, 2, 3)        
        grid_layout.addWidget(QLabel("强度下限:"), 3, 0)
        grid_layout.addWidget(self.edit_I_min, 3, 1)
        grid_layout.addWidget(QLabel("强度上限:"), 3, 2)
        grid_layout.addWidget(self.edit_I_max, 3, 3)    
        grid_layout.addWidget(QLabel("颜色映射:"), 3, 4)   
        grid_layout.addWidget(self.combo_colormap, 3, 5)
        grid_layout.addWidget(QLabel("显示模式:"), 3, 6)
        grid_layout.addWidget(self.combo_display_mode, 3, 7)
        grid_layout.addWidget(QLabel("qxy1下限:"), 4, 0)
        grid_layout.addWidget(self.edit_qxy1_min, 4, 1)
        grid_layout.addWidget(QLabel("qxy1上限:"), 4, 2)
        grid_layout.addWidget(self.edit_qxy1_max, 4, 3)        
        grid_layout.addWidget(QLabel("qz1下限:"), 4, 4)
        grid_layout.addWidget(self.edit_qz1_min, 4, 5)
        grid_layout.addWidget(QLabel("qz1上限:"), 4, 6)
        grid_layout.addWidget(self.edit_qz1_max, 4, 7)        
        grid_layout.addWidget(QLabel("qy2下限:"), 5, 0)
        grid_layout.addWidget(self.edit_qy2_min, 5, 1)
        grid_layout.addWidget(QLabel("qy2上限:"), 5, 2)
        grid_layout.addWidget(self.edit_qy2_max, 5, 3)        
        grid_layout.addWidget(QLabel("qz2下限:"), 5, 4)
        grid_layout.addWidget(self.edit_qz2_min, 5, 5)
        grid_layout.addWidget(QLabel("qz2上限:"), 5, 6)
        grid_layout.addWidget(self.edit_qz2_max, 5, 7)        
        grid_layout.addWidget(QLabel("径向积分角度:"), 0, 4)
        grid_layout.addWidget(self.edit_radi_integ_angle, 0, 5)
        grid_layout.addWidget(QLabel("径向积分范围:"), 0, 6)
        grid_layout.addWidget(self.edit_radi_integ_range, 0, 7)
        grid_layout.addWidget(QLabel("方位积分qc:"), 1, 4)
        grid_layout.addWidget(self.edit_azuth_integ_q, 1, 5)
        grid_layout.addWidget(QLabel("方位积分范围:"), 1, 6)
        grid_layout.addWidget(self.edit_azuth_integ_q_range, 1, 7)


        # 添加常见的15种colormap选项
        colormap_options = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'hot', 'jet', 'gray', 'bone', 'pink', 'gist_ncar',
            'spring', 'summer', 'autumn', 'winter', 'cool',
            'Wistia', 'afmhot', 'gist_heat', 'copper', 'Greys',
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'binary', 'gist_yarg', 'gist_gray', 'nipy_spectral', 
            'gist_heat', 'copper', 'Spectral', 'coolwarm', 
            'bwr', 'seismic', 'twilight', 'turbo', 'rainbow',
            'hsv', 'ocean', 'gist_earth', 'terrain', 'brg',
            'gist_stern', 'CMRmap', 'cubehelix', 'gist_rainbow',                        
        ]
        
        for cmap in colormap_options:
            self.combo_colormap.addItem(cmap)
            
        self.combo_colormap.setCurrentText('hot')           
        
        for col in range(8):
            stretch = 1 if col % 2 == 0 else 2
            grid_layout.setColumnStretch(col, stretch)        
        
        para_layout.addLayout(grid_layout)
        # para_layout.addStretch(1)  # 底部弹性空间
        
        # % 右侧按钮网格
        button_grid_container = QWidget()
        button_right_layout = QGridLayout(button_grid_container)
        self.button_plot_trans = QPushButton("图像翻转")
        self.button_plot_pixel = QPushButton("pixel-pixel")
        self.button_plot_qxy1qz1 = QPushButton("qxy1-qz1")
        self.button_plot_qy2qz2 = QPushButton("qy2-qz2")        
        self.button_radi_see = QPushButton("径向预览")
        self.button_radi_integ = QPushButton("径向积分")
        self.button_azuth_see = QPushButton("方位角预览")
        self.button_azuth_integ = QPushButton("方位角积分")
        button_right_layout.addWidget(self.button_plot_trans,0,0)
        button_right_layout.addWidget(self.button_plot_pixel,0,1)
        button_right_layout.addWidget(self.button_plot_qxy1qz1,0,2)
        button_right_layout.addWidget(self.button_plot_qy2qz2,0,3)        
        button_right_layout.addWidget(self.button_radi_see,1,0)
        button_right_layout.addWidget(self.button_radi_integ,1,1)
        button_right_layout.addWidget(self.button_azuth_see,1,2)
        button_right_layout.addWidget(self.button_azuth_integ,1,3)
        para_layout.addWidget(button_grid_container)
        
        # 右侧容器布局
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.addWidget(para_container)
        right_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # 添加部件到splitter
        bottom_splitter.addWidget(self.canvas1)
        bottom_splitter.addWidget(right_container)
        bottom_splitter.setSizes([900, 100])  # 设置初始分割比例
        
        # 组合主布局
        main_layout.addLayout(button_upper_grid)
        main_layout.addWidget(bottom_splitter, 1)
        self.setCentralWidget(main_widget)
        
        # 数值初始化        
        self.cbar1 = None; self.fig2 = None; self.fig3 = None; self.last_wedge = None; 
        self.default_para_file = "default_parameters.json"
        
        # 构建参数字典映射（用于保存和加载）
        self.parameter_mapping = {
            "centery": self.edit_centery,
            "centerz": self.edit_centerz,
            "distance": self.edit_distance,
            "pixel": self.edit_pixel,
            "lambda_Xray": self.edit_lambda_Xray,
            "incident_angle": self.edit_incident_angle,
            "I_min": self.edit_I_min,
            "I_max": self.edit_I_max,
            "colormap": self.combo_colormap,
            "radi_integ_angle": self.edit_radi_integ_angle,
            "radi_integ_range": self.edit_radi_integ_range,
            "azuth_integ_q": self.edit_azuth_integ_q,
            "azuth_integ_q_range": self.edit_azuth_integ_q_range,
            "qxy1_min": self.edit_qxy1_min,
            "qxy1_max": self.edit_qxy1_max,
            "qz1_min": self.edit_qz1_min,
            "qz1_max": self.edit_qz1_max,
            "qy2_min": self.edit_qy2_min,
            "qy2_max": self.edit_qy2_max,
            "qz2_min": self.edit_qz2_min,
            "qz2_max": self.edit_qz2_max,
        }
        
        # 加载默认参数（在设置硬编码默认值之前）
        self.default_parameters_loaded = load_default_parameters(
            self.default_para_file, self.parameter_mapping)
        
        # 创建结果保存器
        self.saver = ResultSaver(self)
        
        ####################################################################################
        # 只有在没有加载到默认参数时才设置硬编码默认值
        if not hasattr(self, 'default_parameters_loaded') or not self.default_parameters_loaded:            
            self.edit_centery.setValue(797); self.edit_centerz.setValue(-1)
            self.edit_distance.setValue(300); self.edit_pixel.setValue(0.172)
            self.edit_lambda_Xray.setValue(1.18); self.edit_incident_angle.setValue(0.0)
            self.edit_I_min.setValue(1); self.edit_I_max.setValue(1e4)
            self.edit_radi_integ_angle.setValue(90); self.edit_radi_integ_range.setValue(5)
            self.edit_azuth_integ_q.setValue(0.982); self.edit_azuth_integ_q_range.setValue(0.03)
        ####################################################################################        
        
        # 详细功能
        self.btn_load.clicked.connect(self.open_load) # 功能一：导入数据
        self.button_plot_trans.clicked.connect(self.pic_trans) # 功能二：图像翻转
        self.button_plot_pixel.clicked.connect(self.pixel_pixel_plot) # 功能三：绘制pixel图
        self.btn_calibrate.clicked.connect(self.para_calibrate) # 功能四：参数标定
        self.button_plot_qxy1qz1.clicked.connect(self.qxy1_qz1_plot) # 功能五：绘制GIXRS二维图
        self.button_plot_qy2qz2.clicked.connect(self.qy2_qz2_plot) # 功能六：绘制常规qyqz二维图
        self.button_radi_see.clicked.connect(self.radi_see_func) # 功能七：径向积分预览
        self.button_radi_integ.clicked.connect(self.radi_integ_func) # 功能八：径向积分
        self.button_azuth_see.clicked.connect(self.azuth_see_func) # 功能九：方位角积分预览
        self.button_azuth_integ.clicked.connect(self.azuth_integ_func) # 功能十：方位角积分
        self.btn_save.clicked.connect(self.save_pic_txt_func) # 功能十一：保存图像和文本
        self.btn_analysis_batch.clicked.connect(self.batch_func) # 功能十二：批处理
        self.btn_read_poni.clicked.connect(self.read_poni_func) # 功能十三：确定圆心        
        
        # 各个按钮状态管理
        self.btn_load.setEnabled(True); self.btn_analysis_batch.setEnabled(False);         
        self.btn_save.setEnabled(False); self.btn_read_poni.setEnabled(False)
        self.btn_calibrate.setEnabled(False); self.btn_wait_2.setEnabled(False);         
        self.button_plot_trans.setEnabled(False); self.button_plot_pixel.setEnabled(False); 
        self.button_plot_qxy1qz1.setEnabled(False); self.button_plot_qy2qz2.setEnabled(False); 
        self.button_radi_see.setEnabled(False); self.button_radi_integ.setEnabled(False); 
        self.button_azuth_see.setEnabled(False); self.button_azuth_integ.setEnabled(False); 
        
        # 添加按钮提示信息
        self.btn_calibrate.setToolTip("请确认参数区前三行的前两列数值已正确输入!")
        self.button_plot_qxy1qz1.setToolTip("请确认参数区第四和第五行已正确输入!")
        self.button_plot_qy2qz2.setToolTip("请确认参数区第四和第六行已正确输入!")
        self.btn_analysis_batch.setToolTip("使用前务必将右侧参数区填写完整且正确!")
        
        
# 主界面到此结束
############################################################################################################        
        
    # 功能一：导入数据
    # def open_load(self):
    #     # 弹出文件选择对话框
    #     self.file_path = filedialog.askopenfilename(
    #         title='Select a TIFF file', filetypes=[('TIFF files', '*.tif *.tiff'), ('All files', '*.*')])
    #     self.pathName=os.path.dirname(self.file_path)
    #     self.fileName = os.path.basename(self.file_path)
    #     self.pattern = np.array(Image.open(self.file_path))
    #     self.Y, self.Z = np.meshgrid(
    #         sc.arange(0, 1, self.pattern.shape[1]-1), 
    #         sc.arange(0, 1, self.pattern.shape[0]-1)
    #         )
        
    #     # 显示完成提示
    #     QMessageBox.information(self, "数据导入",
    #                             "数据导入已完成", QMessageBox.Ok)
    #     # 激活pixel图绘制
    #     self.button_plot_pixel.setEnabled(True)
    #     result_dict = process_eiger2_gap(self.pattern, extra_gap=extra_gap_value)
    #     self.pattern = result_dict["updated_pattern"]

    # 功能一：导入数据
    def open_load(self):
        # 使用 PyQt5 的文件选择对话框，避免混合 GUI 库
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择 TIFF 文件',
            '',
            'TIFF files (*.tif *.tiff);;All files (*.*)'
        )
        
        # 检查用户是否取消选择
        if not file_path:
            return
        
        try:
            # 处理文件路径
            self.file_path = file_path
            self.pathName = os.path.dirname(self.file_path)
            self.fileName = os.path.basename(self.file_path)
            
            # 加载图像
            self.pattern = np.array(Image.open(self.file_path))
            
            # 创建网格
            self.Y, self.Z = np.meshgrid(
                np.linspace(0, self.pattern.shape[1]-1, self.pattern.shape[1], dtype=int), 
                np.linspace(0, self.pattern.shape[0]-1, self.pattern.shape[0], dtype=int)
            )
            
            # 处理 extra_gap_value（默认值）
            extra_gap_value = 0  # 或根据实际需求设置默认值
            
            # 处理 gap
            result_dict = process_eiger2_gap(self.pattern, extra_gap=extra_gap_value)
            self.pattern = result_dict["updated_pattern"]
            
            # 显示完成提示
            QMessageBox.information(self, "数据导入",
                                    "数据导入已完成", QMessageBox.Ok)
            
            # 激活 pixel 图绘制
            self.button_plot_pixel.setEnabled(True)
            
        except Exception as e:
            # 捕获并显示错误
            QMessageBox.critical(self, "错误", f"数据导入失败: {str(e)}", QMessageBox.Ok)
        
    
    
    # 功能二：图像翻转
    # 功能二：图像翻转（修改后）
    def pic_trans(self):
        if self.pattern is not None:
            dialog = FlipDialog(self, self.pattern)     
            if dialog.exec_() == QDialog.Accepted:
                # 获取旋转后的数据
                self.pattern, self.Y, self.Z = dialog.get_rotated_data()
                
                # 记录翻转信息
                self.flip_status = "已翻转"
                self.flip_dialog = dialog  # 保存对话框实例以获取详细信息
                
                # 显示完成提示
                QMessageBox.information(self, "图像翻转", "图像翻转已完成", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "错误", "请先加载图像", QMessageBox.Ok)
    
    
    # 功能三：绘制pixel图
    def pixel_pixel_plot(self): 
                
        # 清除现有colorbar
        if self.cbar1 is not None:
            self.cbar1.remove()
            self.cbar1 = None
        # 清除axes内容
        current_colormap = self.combo_colormap.currentText()
        
        self.ax1.clear()           
                     
        # 根据选择的显示模式应用不同的颜色映射
        display_mode = self.combo_display_mode.currentText()
        if display_mode == "对数":
            # 使用对数颜色映射
            self.pic1 = self.ax1.pcolormesh(self.Y, self.Z, self.pattern, 
                                            shading='auto', cmap=current_colormap,
                                            norm=LogNorm())
        else:
            # 使用线性颜色映射
            self.pic1 = self.ax1.pcolormesh(self.Y, self.Z, self.pattern, 
                                            shading='auto', cmap=current_colormap)

        self.cbar1 = self.fig1.colorbar(self.pic1, ax=self.ax1, pad=0.02)
        self.ax1.set_xlabel('Y(pixel)', fontsize=12)
        self.ax1.set_ylabel('Z(pixel)', fontsize=12)
        self.ax1.set_aspect('equal')
        self.ax1.set_xlim([np.min(self.Y[0,:]), np.max(self.Y[0,:])])
        self.ax1.set_ylim([np.min(self.Z[:,0]), np.max(self.Z[:,0])])
        self.ax1.set_title(self.fileName)
        
        if (self.edit_I_min.value() < self.edit_I_max.value()):             
            self.pic1.set_clim(self.edit_I_min.value(), self.edit_I_max.value())
            
        # 在 self.ax1（PyQt 画布的子图）上绘制圆心点
        center_y = self.edit_centery.value()
        center_z = self.edit_centerz.value()
        self.ax1.scatter(center_y, center_z, c='r', s=20, marker='o')
        
        self.canvas1.draw()
        # 显示完成提示
        QMessageBox.information(self, "图像绘制",
                                "pixel图像已绘制", QMessageBox.Ok)
        # 激活图像翻转和参数标定
        self.button_plot_trans.setEnabled(True)
        self.btn_read_poni.setEnabled(True)


    # # 功能十三主流版：读取pyfai文件
    # def read_poni_func(self):
    #     poni_path = filedialog.askopenfilename(
    #         title='Select a PONI file', 
    #         filetypes=[('PONI files', '*.poni'), ('All files', '*.*')])
        
    #     para = convert_poni_to_fit2d(poni_path)
    #     self.poni_para = convert_poni_to_fit2d(poni_path)        
        
    #     center_y, center_z = para["center_y"], para["center_z"]
    #     distance, wavelength, pixel = para["distance"], para["wavelength"], para["pixel"]
    #     self.edit_centery.setValue(center_y); self.edit_centerz.setValue(center_z)
    #     self.edit_lambda_Xray.setValue(wavelength)
    #     self.edit_distance.setValue(distance); self.edit_pixel.setValue(pixel)
        
    #     # 显示完成提示
    #     QMessageBox.information(self, "数据保存",
    #                             "计算已完成", QMessageBox.Ok)
    #     self.btn_calibrate.setEnabled(True) 
    #     self.poni_file_path = poni_path


    # 功能十三主流版：读取pyfai文件
    def read_poni_func(self):
        # 使用 PyQt5 的文件选择对话框，避免混合 GUI 库
        poni_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择 PONI 文件',
            '',
            'PONI files (*.poni);;All files (*.*)'
        )
        
        # 检查用户是否取消选择
        if not poni_path:
            return
        
        try:
            # 读取并处理 PONI 文件
            para = convert_poni_to_fit2d(poni_path)
            self.poni_para = para  # 不需要重复调用函数
            
            # 提取参数
            center_y, center_z = para["center_y"], para["center_z"]
            distance, wavelength, pixel = para["distance"], para["wavelength"], para["pixel"]
            
            # 更新界面参数
            self.edit_centery.setValue(center_y)
            self.edit_centerz.setValue(center_z)
            self.edit_lambda_Xray.setValue(wavelength)
            self.edit_distance.setValue(distance)
            self.edit_pixel.setValue(pixel)
            
            # 保存 PONI 文件路径
            self.poni_file_path = poni_path
            
            # 显示完成提示
            QMessageBox.information(self, "数据保存",
                                    "计算已完成", QMessageBox.Ok)
            
            # 激活校准按钮
            self.btn_calibrate.setEnabled(True)
            
        except Exception as e:
            # 捕获并显示错误
            QMessageBox.critical(self, "错误", f"读取 PONI 文件失败: {str(e)}", QMessageBox.Ok)


    
    # 功能四：参数标定
    # 功能四：参数标定（修改后）
    def para_calibrate(self):
        """处理参数标定的UI交互和结果赋值"""
        try:
            # 检查必要数据
            if not hasattr(self, 'Y') or not hasattr(self, 'Z') or not hasattr(self, 'pattern'):
                QMessageBox.warning(self, "错误", "请先加载图像数据", QMessageBox.Ok)
                return
                
            if not hasattr(self, 'poni_para'):
                QMessageBox.warning(self, "错误", "请先读取PONI文件", QMessageBox.Ok)
                return
            
            # 调用独立函数进行计算
            calibration_result = calibrate_parameters(
                self.Y, self.Z, self.pattern, self.poni_para
            )            
            
            # 将结果赋值给实例变量
            self.pattern1 = calibration_result['pattern1']
            self.qxy1 = calibration_result['qxy1']
            self.qz1 = calibration_result['qz1']
            self.q = calibration_result['q']
            self.qy2 = calibration_result['qy2']
            self.qz2 = calibration_result['qz2']
            self.phi = calibration_result['phi'] 
            self.norm_YZ = np.sqrt(calibration_result['ls_Y']**2 + calibration_result['ls_Z']**2)
            
            # 更新UI参数显示范围
            self.edit_qxy1_min.setValue(calibration_result['qxy1_min'])
            self.edit_qxy1_max.setValue(calibration_result['qxy1_max'])
            self.edit_qz1_min.setValue(calibration_result['qz1_min'])
            self.edit_qz1_max.setValue(calibration_result['qz1_max'])
            
            self.edit_qy2_min.setValue(calibration_result['qy2_min'])
            self.edit_qy2_max.setValue(calibration_result['qy2_max'])
            self.edit_qz2_min.setValue(calibration_result['qz2_min'])
            self.edit_qz2_max.setValue(calibration_result['qz2_max'])
            
            # 保存默认参数
            save_default_parameters(self.default_para_file, self.parameter_mapping)
            print("默认参数配置已保存")
            
            # 激活相关功能按钮
            self.button_plot_qxy1qz1.setEnabled(True)
            self.button_plot_qy2qz2.setEnabled(True)
            
            # 显示完成提示
            QMessageBox.information(self, "参数标定", "散射矢量标定已完成", QMessageBox.Ok)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"参数标定过程中出错: {str(e)}", QMessageBox.Ok)
    
    
    # 功能五：绘制GIXRS二维图
    def qxy1_qz1_plot(self):        
        # 检查并关闭已存在的图形窗口
        if hasattr(self, 'fig2'):
            plt.close(self.fig2)
            del self.fig2
        
        current_colormap = self.combo_colormap.currentText()
        
        self.fig2, self.ax2 = plt.subplots(num=2, figsize=(10, 8))
        self.fig2.canvas.manager.set_window_title('qxy-qz Color Map with missing wedge')

        # 根据选择的显示模式应用不同的颜色映射
        display_mode = self.combo_display_mode.currentText()
        if display_mode == "对数":
            # 使用对数颜色映射
            self.pic2 = self.ax2.pcolormesh(self.qxy1, self.qz1, self.pattern1, 
                                            shading='auto', cmap=current_colormap,
                                            norm=LogNorm())
        else:
            # 使用线性颜色映射
            self.pic2 = self.ax2.pcolormesh(self.qxy1, self.qz1, self.pattern1, 
                                            shading='auto', cmap=current_colormap)

        self.cbar2 = self.fig2.colorbar(self.pic2, ax=self.ax2, pad=0.02);
        self.ax2.set_xlabel(r'$q\mathregular{_{xy}\/(\AA^{-1}})$', family="Arial", fontsize=12); 
        self.ax2.set_ylabel(r'$q\mathregular{_{z}\/(\AA^{-1}})$', family="Arial", fontsize=12);
        self.ax2.set_aspect('equal')    
        self.ax2.set_title(self.fileName)
        self.ax2.scatter(0, 0, c='r', s=20, marker='o')
        
        if (self.edit_qxy1_min.value() < self.edit_qxy1_max.value()):             
            self.ax2.set_xlim([self.edit_qxy1_min.value(), self.edit_qxy1_max.value()]);
        if (self.edit_qz1_min.value() < self.edit_qz1_max.value()):             
            self.ax2.set_ylim([self.edit_qz1_min.value(), self.edit_qz1_max.value()]);
        if (self.edit_I_min.value() < self.edit_I_max.value()):             
            self.pic2.set_clim(self.edit_I_min.value(), self.edit_I_max.value());
        
        plt.show(block=False)  # 非阻塞模式，不影响GUI主循环 
        
    
    # 功能六：绘制常规qyqz二维图
    def qy2_qz2_plot(self):        
        # 检查并关闭已存在的图形窗口
        if hasattr(self, 'fig3'):
            plt.close(self.fig3)
            del self.fig3
        
        current_colormap = self.combo_colormap.currentText()
        
        self.fig3, self.ax3 = plt.subplots(num=3, figsize=(10, 8))
        self.fig3.canvas.manager.set_window_title('qy-qz Color Map')

        # 根据选择的显示模式应用不同的颜色映射
        display_mode = self.combo_display_mode.currentText()
        if display_mode == "对数":
            # 使用对数颜色映射
            self.pic3 = self.ax3.pcolormesh(self.qy2, self.qz2, self.pattern, 
                                            shading='auto', cmap=current_colormap,
                                            norm=LogNorm())
        else:
            # 使用线性颜色映射
            self.pic3 = self.ax3.pcolormesh(self.qy2, self.qz2, self.pattern, 
                                            shading='auto', cmap=current_colormap)

        self.cbar3 = self.fig3.colorbar(self.pic3, ax=self.ax3, pad=0.02)
        self.ax3.set_xlabel(r'$q\mathregular{_{y}\/(\AA^{-1}})$', family="Arial", fontsize=12) 
        self.ax3.set_ylabel(r'$q\mathregular{_{z}\/(\AA^{-1}})$', family="Arial", fontsize=12)
        self.ax3.set_aspect('equal')    
        self.ax3.set_title(self.fileName)
        self.ax3.scatter(0, 0, c='r', s=20, marker='o')
        
        if (self.edit_qy2_min.value() < self.edit_qy2_max.value()):             
            self.ax3.set_xlim([self.edit_qy2_min.value(), self.edit_qy2_max.value()])
        if (self.edit_qz2_min.value() < self.edit_qz2_max.value()):             
            self.ax3.set_ylim([self.edit_qz2_min.value(), self.edit_qz2_max.value()])
        if (self.edit_I_min.value() < self.edit_I_max.value()):             
            self.pic3.set_clim(self.edit_I_min.value(), self.edit_I_max.value()) 
            
        plt.show(block=False)  # 非阻塞模式，不影响GUI主循环
        
        # 激活径向积分和单次结果保存
        self.button_radi_see.setEnabled(True)
        self.btn_save.setEnabled(True)
        
       
    # 功能七：径向积分预览
    def radi_see_func(self):
        # 定义阴影区域参数
        center_angle = self.edit_radi_integ_angle.value()   # 中心角度（竖直向上）
        angle_width = self.edit_radi_integ_range.value()    # 总角度宽度（中心左右各10度）
        radius = np.max(self.q)          # 阴影区半径
        
        # 计算扇形边界
        theta1 = center_angle - angle_width  # 起始角度 80度
        theta2 = center_angle + angle_width  # 结束角度 100度        
        
        # 创建扇形补丁
        wedge = mpatches.Wedge(
            center=(0, 0),   # 原点位置
            r=radius,        # 半径
            theta1=theta1,   # 起始角度
            theta2=theta2,    # 结束角度
            alpha=0.2,       # 透明度
            color='yellow',    # 填充颜色
            edgecolor='none',# 边界颜色
            linewidth=1.5)    # 边界线宽
        
        # 移除上次添加的扇形（如果存在）
        if self.last_wedge is not None and self.last_wedge in self.ax3.patches:
            self.last_wedge.remove()
        
        # 添加扇形到坐标轴
        self.ax3.add_patch(wedge)
        self.fig3.canvas.draw_idle() 
        self.last_wedge = wedge
        # 显示完成提示
        # QMessageBox.information(self, "图像预览",
        #                         "径向积分区域已设定", QMessageBox.Ok)
        # 激活径向积分
        self.button_radi_integ.setEnabled(True) 
         
        
    # 功能八：径向积分
    def radi_integ_func(self):
        """径向积分功能"""
        try:
            # 获取参数
            alpha = self.edit_radi_integ_angle.value()
            alpha_range = self.edit_radi_integ_range.value()
            # pixel = self.edit_pixel.value()
            
            alpha_region = (alpha - alpha_range, alpha + alpha_range)
            
            ai = AzimuthalIntegrator()
            ai.load(self.poni_file_path)
            
            # 4. 基本一维积分（最简单用法）
            result_basic = ai.integrate1d(
                data=self.pattern, 
                npt=1001, 
                filename=None, 
                correctSolidAngle=True,
                variance=None, 
                error_model=None,
                radial_range=None, 
                azimuth_range=alpha_region, 
                mask=None, 
                dummy=None, 
                delta_dummy=None,
                polarization_factor=None, 
                dark=None, 
                flat=None, 
                absorption=None,
                method=("bbox", "csr", "cython"), 
                safe=True,
                unit="q_A^-1",
                normalization_factor=1.0,
                metadata=None
                )                          

            # 提取结果
            q_basic = result_basic.radial  # q值
            I_basic = result_basic.intensity  # 强度
            linshi = np.column_stack((q_basic, I_basic))            
            
            # 保存结果并绘图
            self.Im = linshi[linshi[:, 1]>0, :]            
            # 绘图
            self.fig4, self.ax4 = plt.subplots(num=4, figsize=(10, 8))
            self.fig4.canvas.manager.set_window_title('1D I-q integration')
            self.pic4 = self.ax4.plot(self.Im[:, 0], self.Im[:, 1], color='red', linewidth=3)
            self.ax4.set_xlabel(r'$q\mathregular{(\AA^{-1}})$', 
                               family="Arial", fontsize=12, weight='bold')
            self.ax4.set_ylabel('I (a.u.)', family="Arial", fontsize=12, weight='bold')
            self.ax4.set_title(self.fileName, weight='bold')
            self.ax4.set_yscale('log')        
            plt.show(block=False)
            
            # 激活方位角预览
            self.button_azuth_see.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"径向积分过程中出错: {str(e)}", QMessageBox.Ok)
        
    
    # 功能九：方位角预览
    def azuth_see_func(self):
        # 环形积分
        center = (0, 0)      # 圆心坐标
        radius = self.edit_azuth_integ_q.value()         # 平均半径
        width = self.edit_azuth_integ_q_range.value()          # 环宽
        start_angle = 0    # 起始角度（度）
        end_angle = 360      # 终止角度（度）
        wedge = mpatches.Wedge(
                center, 
                radius + width/2,   # 外半径 = 平均半径 + 半宽
                start_angle, 
                end_angle,
                width=width,        # 环宽
                alpha=0.2,          # 透明度设置为30%
                color='yellow',     # 蓝色填充
                edgecolor='none'    # 无边框线
                )
        
        # 移除上次添加的扇形（如果存在）
        if self.last_wedge is not None and self.last_wedge in self.ax3.patches:
            self.last_wedge.remove()        
        
        self.ax3.add_patch(wedge)
        self.fig3.canvas.draw_idle()
        self.last_wedge = wedge 
        
        # 显示完成提示
        # QMessageBox.information(self, "图像预览",
        #                         "径向积分区域已设定", QMessageBox.Ok)
        # 激活方位角积分
        self.button_azuth_integ.setEnabled(True) 
        
        
    # 功能十：方位角积分
    # 功能十：方位角积分（修改后）
    def azuth_integ_func(self):
        """方位角积分功能"""
        try:
            # 获取参数
            q_a = self.edit_azuth_integ_q.value()
            range_q = self.edit_azuth_integ_q_range.value()
            q_region = (q_a - range_q, q_a + range_q)
            
            ai = AzimuthalIntegrator()
            ai.load(self.poni_file_path)
            
            # 4. 基本一维积分（最简单用法）
            result_basic = ai.integrate_radial(
                            data=self.pattern, 
                            npt=721, 
                            npt_rad=31,
                            correctSolidAngle=True,
                            radial_range=q_region, 
                            azimuth_range=(-180, 180),
                            mask=None, 
                            dummy=None, 
                            delta_dummy=None,
                            polarization_factor=None, 
                            dark=None, 
                            flat=None,
                            method=("bbox", "csr", "cython"), 
                            unit="chi_deg", 
                            radial_unit="q_A^-1",
                            normalization_factor=1.0,
                            )
            
            # 提取结果
            q_basic = result_basic.radial  # q值
            I_basic = result_basic.intensity  # 强度            

            # 保存结果并绘图 
            linshi = np.column_stack((q_basic, I_basic))
            
            # 绘图
            self.In = linshi[linshi[:, 1]>0, :]
            self.fig5, self.ax5 = plt.subplots(num=5, figsize=(10, 8))
            self.fig5.canvas.manager.set_window_title('1D I-phi azimuthal scan')
            self.pic5 = self.ax5.plot(self.In[:, 0], self.In[:, 1], color='blue', linewidth=3)
            self.ax5.set_xlabel('phi (deg)', family="Arial", fontsize=12, weight='bold')
            self.ax5.set_ylabel('I (a.u.)', family="Arial", fontsize=12, weight='bold')
            self.ax5.set_title(self.fileName, weight='bold')
            plt.show(block=False)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"方位角积分过程中出错: {str(e)}", QMessageBox.Ok)
        
        
    # 功能十一：保存图像和文本
    # 功能十一：保存图像和文本（修改后）
    def save_pic_txt_func(self):
        """简化后的保存功能 - 只需一行代码"""
        self.saver.save_all_results()
        self.btn_analysis_batch.setEnabled(True)
    
    # 功能十二：批处理    
    # 功能十二：批处理（修改后）
    # def batch_func(self):        
    #     try:
    #         # 检查必要数据
    #         if not hasattr(self, 'pathName') or not self.pathName:
    #             QMessageBox.warning(self, "错误", "请先加载数据文件", QMessageBox.Ok)
    #             return
                
    #         if not hasattr(self, 'poni_file_path'):
    #             QMessageBox.warning(self, "错误", "请先读取PONI文件", QMessageBox.Ok)
    #             return
            
    #         # 收集当前参数
    #         batch_parameters = self._collect_batch_parameters()
            
    #         # 获取所有TIFF文件
    #         all_files = os.listdir(self.pathName)
    #         tiff_files = [
    #             f for f in all_files
    #             if f.lower().endswith(('.tif', '.tiff'))
    #             and os.path.isfile(os.path.join(self.pathName, f))
    #         ]
            
    #         if not tiff_files:
    #             QMessageBox.warning(self, "错误", "未找到TIFF文件", QMessageBox.Ok)
    #             return
            
    #         # 执行批处理
    #         success_count = 0
    #         total_count = len(tiff_files)
            
    #         for i, filename in enumerate(tiff_files):
    #             try:
    #                 result = TwoDpattern_batch_son(
    #                     self.pathName, 
    #                     filename, 
    #                     self.poni_file_path,
    #                     batch_parameters,
    #                 )
                    
    #                 if result:
    #                     success_count += 1
    #                     print(f"处理成功: {filename} ({i+1}/{total_count})")
    #                 else:
    #                     print(f"处理失败: {filename}")
                        
    #             except Exception as e:
    #                 print(f"处理错误 {filename}: {str(e)}")
            
    #         # 显示完成提示
    #         QMessageBox.information(
    #             self, "批处理完成", 
    #             f"批处理完成！成功处理 {success_count}/{total_count} 个文件", 
    #             QMessageBox.Ok
    #         )
            
    #     except Exception as e:
    #         QMessageBox.critical(self, "错误", f"批处理过程中出错: {str(e)}", QMessageBox.Ok)


    def batch_func(self):        
        try:
            # 检查必要数据
            if not hasattr(self, 'pathName') or not self.pathName:
                QMessageBox.warning(self, "错误", "请先加载数据文件", QMessageBox.Ok)
                return
                
            if not hasattr(self, 'poni_file_path'):
                QMessageBox.warning(self, "错误", "请先读取PONI文件", QMessageBox.Ok)
                return
            
            # 收集当前参数
            batch_parameters = self._collect_batch_parameters()
            
            # 获取所有TIFF文件
            all_files = os.listdir(self.pathName)
            tiff_files = [
                f for f in all_files
                if f.lower().endswith(('.tif', '.tiff'))
                and os.path.isfile(os.path.join(self.pathName, f))
            ]
            
            if not tiff_files:
                QMessageBox.warning(self, "错误", "未找到TIFF文件", QMessageBox.Ok)
                return
            
            # 创建进度对话框
            total_count = len(tiff_files)
            progress_dialog = QProgressDialog("批处理准备中...", "取消", 0, total_count, self)
            progress_dialog.setWindowTitle("批处理进度")
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setValue(0)
            
            # 创建并配置批处理线程
            self.batch_thread = BatchProcessThread(
                self.pathName,
                self.poni_file_path,
                tiff_files,
                batch_parameters
            )
            
            # 连接信号槽
            self.batch_thread.progress_updated.connect(lambda value, text: (
                progress_dialog.setValue(value),
                progress_dialog.setLabelText(text)
            ))
            
            self.batch_thread.process_finished.connect(lambda success, total, canceled: (
                progress_dialog.setValue(total),
                progress_dialog.close(),
                self._batch_finished(success, total, canceled)
            ))
            
            self.batch_thread.process_error.connect(lambda error: (
                progress_dialog.close(),
                QMessageBox.critical(self, "错误", f"批处理过程中出错: {error}", QMessageBox.Ok)
            ))
            
            # 连接取消按钮
            progress_dialog.canceled.connect(self.batch_thread.cancel)
            
            # 启动线程
            self.batch_thread.start()
            
            # 显示进度对话框
            progress_dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批处理过程中出错: {str(e)}", QMessageBox.Ok)

    def _batch_finished(self, success_count, total_count, canceled):
        """批处理完成处理"""
        if canceled:
            QMessageBox.information(
                self, "批处理取消", 
                f"批处理已取消。已处理 {success_count}/{total_count} 个文件", 
                QMessageBox.Ok
            )
        else:
            QMessageBox.information(
                self, "批处理完成", 
                f"批处理完成！成功处理 {success_count}/{total_count} 个文件", 
                QMessageBox.Ok
            )


            
    def _collect_batch_parameters(self):
        """收集批处理参数"""
        return {
            "center_y": self.edit_centery.value(),
            "center_z": self.edit_centerz.value(),
            "distance": self.edit_distance.value(),
            "pixel_size": self.edit_pixel.value(),
            "wavelength": self.edit_lambda_Xray.value(),
            "incident_angle": self.edit_incident_angle.value(),
            "I_min": self.edit_I_min.value(),
            "I_max": self.edit_I_max.value(),
            "colormap": self.combo_colormap.currentText(),
            "qxy1_min": self.edit_qxy1_min.value(),
            "qxy1_max": self.edit_qxy1_max.value(),
            "qz1_min": self.edit_qz1_min.value(),
            "qz1_max": self.edit_qz1_max.value(),
            "qy2_min": self.edit_qy2_min.value(),
            "qy2_max": self.edit_qy2_max.value(),
            "qz2_min": self.edit_qz2_min.value(),
            "qz2_max": self.edit_qz2_max.value(),
            "radi_angle": self.edit_radi_integ_angle.value(),
            "radi_range": self.edit_radi_integ_range.value(),
            "azuth_q": self.edit_azuth_integ_q.value(),
            "azuth_range": self.edit_azuth_integ_q_range.value(),
            "flip_mode": getattr(self, 'flip_status', '未翻转'),
            
            "qxy1": self.qxy1,
            "qz1": self.qz1,
            "qy2": self.qy2,
            "qz2": self.qz2,
            "poni_para": self.poni_para,
            
            "Y": self.Y,
            "Z": self.Z,
            }

 
# %% 11. 批处理函数
# 批处理子函数（修改后）
def TwoDpattern_batch_son(pathName, current_filename, poni_path, batch_parameters):
    """
    批处理子函数 - 处理单个文件
    
    参数:
        pathName: 文件路径
        current_filename: 文件名
        batch_parameters: 批处理参数字典
        poni_para: PONI参数
        colormap_mode: 颜色映射模式
        
    返回:
        bool: 处理是否成功
    """
    try:
        # 提取参数
        qxy1_min = batch_parameters["qxy1_min"]; qxy1_max = batch_parameters["qxy1_max"]
        qz1_min = batch_parameters["qz1_min"]; qz1_max = batch_parameters["qz1_max"]
        
        qy2_min = batch_parameters["qy2_min"]; qy2_max = batch_parameters["qy2_max"]
        qz2_min = batch_parameters["qz2_min"]; qz2_max = batch_parameters["qz2_max"]
        
        radi_angle = batch_parameters["radi_angle"]; radi_range = batch_parameters["radi_range"]
        azuth_q = batch_parameters["azuth_q"]; azuth_range = batch_parameters["azuth_range"]; 
        
        I_min = batch_parameters["I_min"]; I_max = batch_parameters["I_max"]   
        qxy1 = batch_parameters["qxy1"]; qz1 = batch_parameters["qz1"]
        qy2 = batch_parameters["qy2"]; qz2 = batch_parameters["qz2"]
        
        Y = batch_parameters["Y"]; Z = batch_parameters["Z"]
        
        poni_para = batch_parameters["poni_para"]
        current_colormap = batch_parameters["colormap"]
        
        
        # 功能一：读取TIFF文件
        file_path = os.path.join(pathName, current_filename)
        pattern0 = np.array(Image.open(file_path))     
        result_dict = process_eiger2_gap(pattern0, extra_gap=extra_gap_value)
        pattern0 = result_dict["updated_pattern"]
        
        
        # 功能二：图像翻转（简化版，不进行交互式翻转）
        pattern = pattern0.copy()  # 默认不翻转
        
        # 功能三：参数标定（使用独立函数）
        calibration_result = calibrate_parameters(Y, Z, pattern, poni_para)
        if calibration_result is None:
            return False
        
        pattern1 = calibration_result['pattern1']
        
        
        # 创建分析文件夹
        baseName = os.path.splitext(current_filename)[0]
        analysis_folder = os.path.join(pathName, f"{baseName}_analysis")
        os.makedirs(analysis_folder, exist_ok=True)
        
        # 功能四：绘制pixel图
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        fig1.canvas.manager.set_window_title('2D pixel Color Map')
        pic1 = ax1.pcolormesh(Y, Z, pattern, shading='auto', cmap=current_colormap)
        fig1.colorbar(pic1, ax=ax1, pad=0.02)
        ax1.set_xlabel('Y(pixel)', fontsize=12)
        ax1.set_ylabel('Z(pixel)', fontsize=12)
        ax1.set_aspect('equal')
        ax1.set_title(current_filename)
        pic1.set_clim(I_min, I_max)
        
        # # 功能五：绘制GIXRS二维图
        # fig2, ax2 = plt.subplots(figsize=(10, 8))
        # fig2.canvas.manager.set_window_title('qxy-qz Color Map with missing wedge')
        # pic2 = ax2.pcolormesh(qxy1, qz1, pattern1, shading='auto', cmap=current_colormap)
        # fig2.colorbar(pic2, ax=ax2, pad=0.02)
        # ax2.set_xlabel(r'$q\mathregular{_{xy}\/(\AA^{-1}})$', family="Arial", fontsize=12)
        # ax2.set_ylabel(r'$q\mathregular{_{z}\/(\AA^{-1}})$', family="Arial", fontsize=12)
        # ax2.set_aspect('equal')
        # ax2.set_title(current_filename)
        # ax2.set_xlim(qxy1_min, qxy1_max)
        # ax2.set_ylim(qz1_min, qz1_max)
        # pic2.set_clim(I_min, I_max)
        
        # # 功能六：绘制常规qyqz二维图
        # fig3, ax3 = plt.subplots(figsize=(10, 8))
        # fig3.canvas.manager.set_window_title('qy-qz Color Map')
        # pic3 = ax3.pcolormesh(qy2, qz2, pattern, shading='auto', cmap=current_colormap)
        # fig3.colorbar(pic3, ax=ax3, pad=0.02)
        # ax3.set_xlabel(r'$q\mathregular{_{y}\/(\AA^{-1}})$', family="Arial", fontsize=12)
        # ax3.set_ylabel(r'$q\mathregular{_{z}\/(\AA^{-1}})$', family="Arial", fontsize=12)
        # ax3.set_aspect('equal')
        # ax3.set_title(current_filename)
        # ax3.set_xlim(qy2_min, qy2_max)
        # ax3.set_ylim(qz2_min, qz2_max)
        # pic3.set_clim(I_min, I_max)
        
        
        # 功能七：径向积分
        # 4. 基本一维积分（最简单用法）
        ai = AzimuthalIntegrator()
        ai.load(poni_path)
        radi_region = (radi_angle - radi_range, radi_angle + radi_range)        
        
        result_basic = ai.integrate1d(
            data=pattern, 
            npt=1001, 
            filename=None, 
            correctSolidAngle=True,
            variance=None, 
            error_model=None,
            radial_range=None, 
            azimuth_range=radi_region, 
            mask=None, 
            dummy=None, 
            delta_dummy=None,
            polarization_factor=None, 
            dark=None, 
            flat=None, 
            absorption=None,
            method=("bbox", "csr", "cython"), 
            safe=True,
            unit="q_A^-1",
            normalization_factor=1.0,
            metadata=None
            )
                        

        # 提取结果
        q_basic = result_basic.radial  # q值
        I_basic = result_basic.intensity  # 强度
        linshi = np.column_stack((q_basic, I_basic))
        Im = linshi[linshi[:, 1]>0, :]        
       
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        fig4.canvas.manager.set_window_title('1D I-q integration')
        ax4.plot(Im[:, 0], Im[:, 1], color='red', linewidth=3)
        ax4.set_xlabel(r'$q\mathregular{_{y}\/(\AA^{-1}})$', 
                      family="Arial", fontsize=12, weight='bold')
        ax4.set_ylabel('I (a.u.)', family="Arial", fontsize=12, weight='bold')
        ax4.set_title(current_filename, weight='bold')
        ax4.set_yscale('log')
        
        
        # # 功能八：方位角积分
        # q_region = (azuth_q - azuth_range, azuth_q + azuth_range)
        
        # # 4. 基本一维积分（最简单用法）
        # result_basic = ai.integrate_radial(
        #                 data=pattern, 
        #                 npt=721, 
        #                 npt_rad=31,
        #                 correctSolidAngle=True,
        #                 radial_range=q_region, 
        #                 azimuth_range=(-180, 180),
        #                 mask=None, 
        #                 dummy=None, 
        #                 delta_dummy=None,
        #                 polarization_factor=None, 
        #                 dark=None, 
        #                 flat=None,
        #                 method=("bbox", "csr", "cython"),  
        #                 unit="chi_deg", 
        #                 radial_unit="q_A^-1",
        #                 normalization_factor=1.0,
        #                 )
        
        # # 提取结果
        # chi_basic = result_basic.radial  # q值
        # I_basic = result_basic.intensity  # 强度
        # linshi = np.column_stack((chi_basic, I_basic))
        # In = linshi[linshi[:, 1]>0, :]
        
        # fig5, ax5 = plt.subplots(figsize=(10, 8))
        # fig5.canvas.manager.set_window_title('1D I-phi azimuthal scan')
        # ax5.plot(In[:, 0], In[:, 1], color='blue', linewidth=3)
        # ax5.set_xlabel('phi (deg)', family="Arial", fontsize=12, weight='bold')
        # ax5.set_ylabel('I (a.u.)', family="Arial", fontsize=12, weight='bold')
        # ax5.set_title(current_filename, weight='bold')
        
        
        # 功能九：保存所有结果
        # 保存图像
        if 'fig1' in locals():
            fig1.savefig(os.path.join(analysis_folder, f"{baseName}_pixel_2D_pattern.png"), 
                        dpi=200, bbox_inches='tight')
        
        # if 'fig2' in locals():
        #     fig2.savefig(os.path.join(analysis_folder, f"{baseName}_qz-qxy_2D_pattern.png"), 
        #                 dpi=200, bbox_inches='tight')
        
        # if 'fig3' in locals():
        #     fig3.savefig(os.path.join(analysis_folder, f"{baseName}_qz-qy_2D_pattern.png"), 
        #                 dpi=200, bbox_inches='tight')
        
        if 'fig4' in locals():
            fig4.savefig(os.path.join(analysis_folder, f"{baseName}_I-q_1D_integration.png"), 
                        dpi=200, bbox_inches='tight')
        
        # if 'fig5' in locals():
        #     fig5.savefig(os.path.join(analysis_folder, f"{baseName}_I-phi_1D_integration.png"), 
        #                 dpi=200, bbox_inches='tight')
        
        # 保存文本数据
        if 'Im' in locals():
            np.savetxt(os.path.join(analysis_folder, f"{baseName}_I-q_data.txt"), 
                      Im, fmt='%.6f', delimiter='\t')
        
        # if 'In' in locals():
        #     np.savetxt(os.path.join(analysis_folder, f"{baseName}_I-phi_data.txt"), 
        #               In, fmt='%.6f', delimiter='\t')
        
        # 保存原始图像
        Image.fromarray(pattern).save(os.path.join(analysis_folder, f"{baseName}.tif"))        
       
        # 关闭所有图形
        plt.close('all')        
        return True
        
    except Exception as e:
        print(f"批处理子函数错误 {current_filename}: {str(e)}")
        # 确保关闭所有图形
        plt.close('all')
        return False


# %% 显示窗口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TwoDpattern()
    window.show()
    sys.exit(app.exec_())
    
    