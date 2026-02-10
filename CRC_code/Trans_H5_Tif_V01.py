import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QSpinBox, QPushButton, 
                             QTextEdit, QFileDialog, QMessageBox, QProgressBar,
                             QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from Tool.read_h5_to_dict import (get_all_files_in_folder, batch_convert_h5_to_tif)


class ConversionThread(QThread):
    """批量转换线程"""
    progress_updated = pyqtSignal(int, str)
    conversion_complete = pyqtSignal(dict)
    conversion_error = pyqtSignal(str)
    
    def __init__(self, folder_path, filter_str, channel):
        super().__init__()
        self.folder_path = folder_path
        self.filter_str = filter_str
        self.channel = channel
        self.is_running = True
    
    def run(self):
        try:
            # 步骤1: 获取文件列表
            self.progress_updated.emit(10, "正在扫描文件夹...")
            all_data_file = get_all_files_in_folder(
                folder_path=self.folder_path, 
                filter_str=self.filter_str
            )
            
            if not all_data_file.get('filtered_h5_files'):
                self.conversion_error.emit(f"未找到包含 '{self.filter_str}' 的HDF5文件")
                return
            
            file_count = len(all_data_file['filtered_h5_files'])
            self.progress_updated.emit(30, f"找到 {file_count} 个文件，开始转换...")
            
            # 步骤2: 批量转换
            batch_results = batch_convert_h5_to_tif(
                all_data_file, 
                channel_selected=self.channel
            )
            
            self.progress_updated.emit(100, "转换完成！")
            self.conversion_complete.emit(batch_results)
            
        except Exception as e:
            self.conversion_error.emit(f"转换过程中出错: {str(e)}")


class H5BatchConverterGUI(QMainWindow):
    """HDF5批量转TIFF独立GUI界面"""
    
    def __init__(self):
        super().__init__()
        self.conversion_thread = None
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("HDF5批量转换工具 v1.0")
        self.setGeometry(100, 100, 600, 500)
        
        # 设置窗口图标
        try:
            self.setWindowIcon(QIcon("icon.png"))
        except:
            pass
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("HDF5文件批量转TIFF工具")
        title_font = QFont("Arial", 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # 参数设置组
        settings_group = QGroupBox("转换参数设置")
        settings_layout = QGridLayout()
        
        # 1. 文件名特征
        settings_layout.addWidget(QLabel("文件名特征:"), 0, 0)
        self.filter_input = QLineEdit("data_000001.h5")
        self.filter_input.setPlaceholderText("请输入文件名特征，如: data_000001.h5")
        settings_layout.addWidget(self.filter_input, 0, 1, 1, 2)
        
        # 添加说明
        filter_note = QLabel("提示: 输入文件名中的特征字符串，程序将筛选包含此字符串的文件")
        filter_note.setStyleSheet("color: #666; font-size: 10px;")
        settings_layout.addWidget(filter_note, 1, 0, 1, 3)
        
        # 2. 通道选择
        settings_layout.addWidget(QLabel("选择通道:"), 2, 0)
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 10)
        self.channel_spin.setValue(0)
        self.channel_spin.setToolTip("选择要转换的通道编号（0对应第一个通道）")
        settings_layout.addWidget(self.channel_spin, 2, 1)
        
        channel_note = QLabel("通道0: 第一个数据通道")
        channel_note.setStyleSheet("color: #666; font-size: 10px;")
        settings_layout.addWidget(channel_note, 2, 2)
        
        # 3. 文件夹选择
        settings_layout.addWidget(QLabel("选择总文件夹:"), 3, 0)
        
        self.folder_path_input = QLineEdit()
        self.folder_path_input.setReadOnly(True)
        self.folder_path_input.setPlaceholderText("请选择包含HDF5文件的文件夹")
        settings_layout.addWidget(self.folder_path_input, 3, 1)
        
        self.select_folder_btn = QPushButton("浏览...")
        self.select_folder_btn.clicked.connect(self.select_folder)
        settings_layout.addWidget(self.select_folder_btn, 3, 2)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # 进度条
        main_layout.addWidget(QLabel("转换进度:"))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #2c3e50; padding: 5px;")
        main_layout.addWidget(self.status_label)
        
        # 日志输出
        main_layout.addWidget(QLabel("转换日志:"))
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        main_layout.addWidget(self.log_output)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始批量转换")
        self.start_btn.clicked.connect(self.start_conversion)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        # 设置整体样式
        central_widget.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #2c3e50;
            }
            QLineEdit, QSpinBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
            }
        """)
    
    def select_folder(self):
        """选择文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "选择包含HDF5文件的文件夹"
        )
        if folder_path:
            self.folder_path_input.setText(folder_path)
            self.log_message(f"已选择文件夹: {folder_path}")
    
    def log_message(self, message):
        """添加日志消息"""
        self.log_output.append(f"> {message}")
    
    def start_conversion(self):
        """开始批量转换"""
        # 获取输入参数
        folder_path = self.folder_path_input.text()
        filter_str = self.filter_input.text()
        channel = self.channel_spin.value()
        
        # 验证输入
        if not folder_path or not os.path.exists(folder_path):
            QMessageBox.warning(self, "警告", "请先选择有效的文件夹路径")
            return
        
        if not filter_str.strip():
            QMessageBox.warning(self, "警告", "请输入文件名特征")
            return
        
        # 显示确认对话框
        reply = QMessageBox.question(
            self, 
            "确认", 
            f"将转换文件夹 '{os.path.basename(folder_path)}' 中所有包含 '{filter_str}' 的HDF5文件\n\n" +
            f"保存通道: {channel}\n\n" +
            "是否继续?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 重置界面状态
        self.progress_bar.setValue(0)
        self.status_label.setText("正在启动转换...")
        self.log_message("=" * 50)
        self.log_message(f"开始批量转换")
        self.log_message(f"文件夹: {folder_path}")
        self.log_message(f"文件名特征: {filter_str}")
        self.log_message(f"保存通道: {channel}")
        
        # 禁用开始按钮，启用取消按钮
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # 创建并启动转换线程
        self.conversion_thread = ConversionThread(folder_path, filter_str, channel)
        self.conversion_thread.progress_updated.connect(self.update_progress)
        self.conversion_thread.conversion_complete.connect(self.conversion_finished)
        self.conversion_thread.conversion_error.connect(self.conversion_failed)
        self.conversion_thread.start()
    
    def update_progress(self, value, message):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.log_message(message)
    
    def conversion_finished(self, results):
        """转换完成"""
        self.progress_bar.setValue(100)
        self.status_label.setText("转换完成！")
        
        # 显示结果
        success = results.get('success', 0)
        total = results.get('total', 0)
        output_dir = results.get('output_dir', '')
        
        self.log_message(f"转换完成！")
        self.log_message(f"成功转换: {success}/{total} 个文件")
        self.log_message(f"输出目录: {output_dir}")
        
        # 询问是否打开输出文件夹
        if success > 0 and os.path.exists(output_dir):
            reply = QMessageBox.question(
                self,
                "转换完成",
                f"成功转换 {success} 个文件！\n\n是否打开输出文件夹？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.open_folder(output_dir)
        
        # 重置按钮状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
    
    def conversion_failed(self, error_message):
        """转换失败"""
        self.status_label.setText("转换失败")
        self.log_message(f"错误: {error_message}")
        QMessageBox.critical(self, "转换失败", error_message)
        
        # 重置按钮状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
    
    def cancel_conversion(self):
        """取消转换"""
        if self.conversion_thread and self.conversion_thread.isRunning():
            self.conversion_thread.is_running = False
            self.conversion_thread.terminate()
            self.conversion_thread.wait()
            
            self.status_label.setText("转换已取消")
            self.log_message("用户取消了转换")
            
            # 重置按钮状态
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
    
    def open_folder(self, folder_path):
        """打开文件夹"""
        try:
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", folder_path])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", folder_path])
        except Exception as e:
            self.log_message(f"无法打开文件夹: {str(e)}")
    
    def closeEvent(self, event):
        """关闭窗口事件"""
        if self.conversion_thread and self.conversion_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "转换正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cancel_conversion()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建并显示窗口
    window = H5BatchConverterGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()