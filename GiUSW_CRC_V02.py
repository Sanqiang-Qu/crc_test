import sys
import os
import time

from PyQt5.QtWidgets import (QApplication, QStackedWidget, QMainWindow, 
                             QAction, QToolBar, QWidget,
                            QStatusBar, QLabel, QVBoxLayout, QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt, QProcess, QTimer

# 添加Single_tool目录到Python路径
CRC_code_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CRC_code")
if CRC_code_path not in sys.path:
    sys.path.insert(0, CRC_code_path)

from CRC_code.TwoDpattern_V02 import TwoDpattern
from CRC_code.Trans_H5_Tif_V01 import H5BatchConverterGUI

# Windows窗口操作库（用于嵌入外部窗口）
import win32gui
import win32con
import win32process



# ------------------------------
# PyFAI校准模块（嵌入pyfai-calib2）
# ------------------------------
class PyFAICalibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None       # pyfai-calib2进程对象
        self.child_hwnd = None    # 存储pyfai窗口句柄
        self.init_ui()
        

    def init_ui(self):
        """初始化界面布局"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # 去除边距
        # 初始显示提示文本
        self.placeholder = QLabel("点击上方「PyFAI 校准」按钮启动校准工具")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.placeholder)

    def start_pyfai(self):
        """启动pyfai-calib2进程并嵌入窗口"""
        # 移除占位文本
        if self.placeholder:
            self.layout.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None

        # 若进程已运行，先停止
        if self.process and self.process.state() == QProcess.Running:
            self.stop_pyfai()

        # 启动新进程
        self.process = QProcess(self)
        self.process.started.connect(self._on_process_started)
        self.process.finished.connect(self._on_process_finished)
        try:
            self.process.start("pyfai-calib2")  # 启动命令
        except Exception as e:
            QMessageBox.critical(self, "启动失败", f"无法启动pyfai-calib2：{str(e)}")
            self._restore_placeholder()

    def stop_pyfai(self):
        """停止pyfai-calib2进程并清理窗口"""
        # 终止进程
        if self.process and self.process.state() == QProcess.Running:
            self.process.terminate()
            self.process.waitForFinished(1000)  # 等待1秒
        self.process = None

        # 移除嵌入的窗口
        if self.child_hwnd:
            win32gui.SetParent(self.child_hwnd, None)  # 解除父窗口关联
            win32gui.ShowWindow(self.child_hwnd, win32con.SW_HIDE)  # 隐藏窗口
            self.child_hwnd = None

        # 恢复占位文本
        self._restore_placeholder()

    def _restore_placeholder(self):
        """恢复初始占位文本"""
        if not self.placeholder and not self.child_hwnd:
            self.placeholder = QLabel("点击上方「PyFAI 校准」按钮启动校准工具")
            self.placeholder.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.placeholder)

    def _on_process_started(self):
        """进程启动后，查找并嵌入窗口"""
        def find_pyfai_window(hwnd, result):
            """枚举窗口的回调函数，寻找PyFAI主窗口"""
            if win32gui.IsWindowVisible(hwnd) and "PyFAI Calibration" in win32gui.GetWindowText(hwnd):
                result[0] = hwnd
                return False  # 找到后停止枚举
            return True

        # 枚举所有窗口查找PyFAI
        hwnd_result = [None]
        win32gui.EnumWindows(find_pyfai_window, hwnd_result)
        self.child_hwnd = hwnd_result[0]

        if self.child_hwnd:
            # 验证窗口属于当前进程
            _, process_id = win32process.GetWindowThreadProcessId(self.child_hwnd)
            if process_id == self.process.processId():
                # 嵌入当前窗口
                win32gui.SetParent(self.child_hwnd, self.winId())
                win32gui.ShowWindow(self.child_hwnd, win32con.SW_SHOW)
                self.resizeEvent(None)  # 调整大小
            else:
                QMessageBox.warning(self, "窗口匹配失败", "未找到PyFAI校准窗口，请重试")
                self.stop_pyfai()
        else:
            QMessageBox.warning(self, "窗口未找到", "正开启新的PyFAI校准窗口")
            self.stop_pyfai()

    def _on_process_finished(self, exit_code, exit_status):
        """进程结束时清理"""
        self.child_hwnd = None
        self._restore_placeholder()
        print(f"PyFAI进程已结束，退出码: {exit_code}")

    def resizeEvent(self, event):
        """窗口大小变化时同步调整嵌入窗口"""
        if self.child_hwnd:
            # 调整子窗口大小以填充当前控件
            win32gui.MoveWindow(self.child_hwnd, 0, 0, self.width(), self.height(), True)
        super().resizeEvent(event)

    def closeEvent(self, event):
        """窗口关闭时确保进程终止和定时器停止"""
        # 停止运行时间定时器
        if hasattr(self, 'running_timer'):
            self.running_timer.stop()
        
        # 停止PyFAI进程（如果正在运行）
        pyfai_module = self.get_module_instance("PyFAI 校准")
        if pyfai_module:
            pyfai_module.stop_pyfai()
        
        super().closeEvent(event)

# ------------------------------
# 主应用程序类
# ------------------------------
class MainApplication(QMainWindow):
    current_module_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("算法平台 - 多模块集成系统")
        self.resize(1600, 800)  # 初始窗口大小
        
        # 记录启动时间
        self.start_time = time.time()
        
        # 初始化状态栏
        self.init_status_bar()
        
        # 创建堆叠窗口（用于模块切换）
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 模块管理
        self.modules = {}  # 存储模块：{名称: {实例, 描述}}
        self.module_index_map = {}  # 模块名称->索引映射
        
        # 初始化菜单和工具栏
        self.init_menu_bar()
        self.init_tool_bar()
        
        # 加载所有模块
        self.load_modules()
        
        # 默认显示"二维图和一维曲线"模块
        self.show_default_module()
        
        # 新增：启动定时器，每隔60秒输出运行时间
        self.setup_running_timer()
        

    def setup_running_timer(self):
        """设置运行时间定时器"""
        # 创建定时器
        self.running_timer = QTimer()
        self.running_timer.timeout.connect(self.print_running_time)
        self.running_timer.start(60000 * 5)  # 60秒 = 60000毫秒
        
        # 立即输出一次
        self.print_running_time()        
        
    def print_running_time(self):
        """输出运行时间到终端"""
        elapsed_time = int(time.time() - self.start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        print(f"[主界面运行时间] GUI界面已运行: {minutes}分{seconds}秒")
        sys.stdout.flush()  # 确保立即输出
        


    def show_default_module(self):
        """显示默认模块（二维图和一维曲线）"""
        # 检查"二维图和一维曲线"模块是否存在
        if "二维图和一维曲线" in self.module_index_map:
            self.switch_module("二维图和一维曲线")
        elif self.modules:  # 如果有其他模块，显示第一个
            first_module = next(iter(self.modules.keys()))
            self.switch_module(first_module)
        else:
            # 如果没有模块，显示错误信息
            self.status_label.setText("错误：没有可用的功能模块")

    # ------------------------------
    # 初始化界面组件
    # ------------------------------
    def init_status_bar(self):
        """初始化状态栏"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("就绪")
        self.statusBar.addWidget(self.status_label)

    def init_menu_bar(self):
        """初始化菜单栏"""
        # 模块菜单
        self.module_menu = self.menuBar().addMenu("模块")
        
        # 帮助菜单
        help_menu = self.menuBar().addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def init_tool_bar(self):
        """初始化工具栏"""
        self.module_toolbar = QToolBar("模块切换")
        self.module_toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(self.module_toolbar)


    ##########################################################################
    # ------------------------------
    # 模块管理
    # ------------------------------
    def load_modules(self):
        """加载所有功能模块（按顺序添加）"""
        # 1. PyFAI校准模块（第一个模块）
        self.add_module(
            module_name="PyFAI 校准",
            module_instance=PyFAICalibWidget(),
            description="使用pyfai-calib2进行衍射校准"
        )
        
        # 2. 二维图和一维曲线模块（请根据实际路径启用）
        try:            
            self.add_module(
                module_name="二维图和一维曲线",
                module_instance=TwoDpattern(),
                description="数据读取、标定、归一化和预处理结果保存"
            )
        except ImportError as e:
            print(f"警告：未找到二维图模块 - {e}")
            
        # 3. H5批量转TIF模块    
        try:            
            self.add_module(
                module_name="H5批量转TIF",
                module_instance=H5BatchConverterGUI(),
                description="批量将HDF5文件转换为TIFF格式"
            )
        except ImportError as e:
            print(f"警告：未找到H5批量转换模块 - {e}")
        
        
        # # 4. SAXS尺寸分布模块（请根据实际路径启用）
        # try:            
        #     self.add_module(
        #         module_name="SAXS尺寸分布",
        #         module_instance=SAXSApp(),
        #         description="读取预处理数据，进行粒子群优化计算并保存结果"
        #     )
        # except ImportError as e:
        #     print(f"警告：未找到SAXS模块 - {e}")
            
        
        # # 5. XGBoost机器学习模块（新增第四个模块）
        # try:
        #     self.add_module(
        #         module_name="XGBoost机器学习",
        #         module_instance=XGBoostApp(),
        #         description="使用XGBoost进行散射曲线拟合和尺寸分布预测"
        #     )
        # except Exception as e:
        #     print(f"警告：XGBoost模块初始化失败 - {e}")
##############################################################################    
    

    def add_module(self, module_name, module_instance, description=""):
        """添加模块到系统"""
        if not isinstance(module_instance, QWidget):
            raise ValueError("模块必须是QWidget子类")
        
        if module_name in self.modules:
            print(f"警告：模块'{module_name}'已存在，将被替换")
        
        # 存储模块信息
        self.modules[module_name] = {
            "instance": module_instance,
            "description": description
        }
        
        # 添加到堆叠窗口
        index = self.stacked_widget.addWidget(module_instance)
        self.module_index_map[module_name] = index
        
        # 添加菜单动作
        menu_action = QAction(module_name, self)
        menu_action.triggered.connect(
            lambda checked, name=module_name: self.switch_module(name)
        )
        menu_action.setStatusTip(description)
        self.module_menu.addAction(menu_action)
        
        # 添加工具栏按钮
        toolbar_action = QAction(module_name, self)
        toolbar_action.triggered.connect(
            lambda checked, name=module_name: self.switch_module(name)
        )
        toolbar_action.setStatusTip(description)
        self.module_toolbar.addAction(toolbar_action)
        

    def switch_module(self, module_name):
        """切换到指定模块，控制PyFAI进程"""
        # 切换前：如果离开PyFAI模块，停止其进程
        if module_name != "PyFAI 校准":
            pyfai_module = self.get_module_instance("PyFAI 校准")
            if pyfai_module:
                pyfai_module.stop_pyfai()
        
        # 验证模块存在性
        if module_name not in self.module_index_map:
            QMessageBox.warning(self, "错误", f"模块'{module_name}'不存在")
            return
        
        # 切换堆叠窗口
        self.stacked_widget.setCurrentIndex(self.module_index_map[module_name])
        
        # 更新状态栏
        desc = self.modules[module_name]["description"]
        self.status_label.setText(f"当前模块：{module_name} - {desc}")
        self.current_module_changed.emit(module_name)
        
        # 切换后：如果进入PyFAI模块，启动其进程
        if module_name == "PyFAI 校准":
            pyfai_module = self.get_module_instance("PyFAI 校准")
            if pyfai_module:
                pyfai_module.start_pyfai()

    def get_module_instance(self, module_name):
        """获取模块实例"""
        return self.modules.get(module_name, {}).get("instance")

    # ------------------------------
    # 其他功能
    # ------------------------------
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, 
            "关于", 
            "算法平台 - 多模块集成系统\n版本 1.0\n\n集成PyFAI校准、二维图分析等功能"
        )

# ------------------------------
# 程序入口
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置应用样式（可选）
    app.setStyle("Fusion")  # 跨平台统一样式
    main_window = MainApplication()
    main_window.show()
    sys.exit(app.exec_())
    