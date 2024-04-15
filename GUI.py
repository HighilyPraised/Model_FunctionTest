import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, \
    QProgressBar, QSizePolicy, QLineEdit, QMainWindow
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from Model import StyleTransferModel
import time

Model_Path = './trained/pretrained_model.pth'
Image_Root = './images/'
Cache_Root = './cache/'
Output_Root = './output/'

class StyleTransferThread(QThread):
    progress_signal = pyqtSignal(int)
    def __init__(self, model, input_image_path, output_image_path):
        super().__init__()
        self.model = model
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
    def run(self):
        # 这里假设你的模型有一个接受进度回调的参数
        self.model.style_transfer(self.input_image_path, self.output_image_path, self.update_progress)
    def update_progress(self, progress):
        self.progress_signal.emit(progress)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        # # 设置窗口初始大小
        # self.resize(800, 600)
        # # 创建垂直布局
        # layout = QVBoxLayout()
        # self.setLayout(layout)
        # # 创建"选择图片"按钮
        # btn = QPushButton('Select Picture', self)
        # btn.clicked.connect(self.openFileNameDialog)
        # layout.addWidget(btn)
        # # 创建进度条
        # self.progress = QProgressBar(self)
        # self.progress.setFormat('%v/%m')
        # self.progress.setAlignment(Qt.AlignCenter)  # 文字居中对齐
        # layout.addWidget(self.progress)
        # self.progress.setDisabled(True)
        # # 创建水平布局用于显示图片
        # hlayout = QHBoxLayout()
        # layout.addLayout(hlayout)
        # # 创建标签用于显示原始图片
        # self.label_input = QLabel(self)
        # self.label_input.setAlignment(Qt.AlignCenter)
        # self.label_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # hlayout.addWidget(self.label_input)
        # # 创建标签用于显示输出图片
        # self.label_output = QLabel(self)
        # self.label_output.setAlignment(Qt.AlignCenter)
        # self.label_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # hlayout.addWidget(self.label_output)

        self.setMinimumSize(800, 600)
        self.setMaximumSize(1600, 1050)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # 创建左侧和右侧的layout
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # 设置左侧layout的size policy为固定，宽度为窗口宽度的25%
        self.left_layout.setSizeConstraint(QVBoxLayout.SetNoConstraint)
        self.left_layout_widget = QWidget()
        self.left_layout_widget.setLayout(self.left_layout)
        self.left_layout_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.left_layout_widget.setFixedWidth(int(self.width() * 0.25))

        # 添加按钮和输入框到左侧layout
        self.open_image_button = QPushButton('Open Image')
        self.open_image_button.clicked.connect(self.openFileNameDialog)
        self.open_video_button = QPushButton('Open Video')
        self.parameter1_input = QLineEdit('512')
        self.parameter2_input = QLineEdit('0.5')
        self.input_layout = QHBoxLayout()
        self.input_layout.addWidget(self.parameter1_input)
        self.input_layout.addWidget(self.parameter2_input)
        self.left_layout.addWidget(self.open_image_button)
        self.left_layout.addWidget(self.open_video_button)
        self.left_layout.addLayout(self.input_layout)
        # 设置右侧layout的size policy为可扩展，宽度为窗口宽度的75%
        self.right_layout.setSizeConstraint(QVBoxLayout.SetDefaultConstraint)
        self.right_layout_widget = QWidget()
        self.right_layout_widget.setLayout(self.right_layout)
        self.right_layout_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 添加标签和进度条到右侧layout
        self.label_input = QLabel()
        self.progress = QProgressBar()
        self.progress.setFormat('%v/%m')
        self.progress.setAlignment(Qt.AlignCenter)  # 文字居中对齐
        self.label_output = QLabel()
        self.right_layout.addWidget(self.label_input)
        self.right_layout.addWidget(self.progress)
        self.right_layout.addWidget(self.label_output)

        # 设置进度条的高度为最小值，两个label平分剩下的高度
        self.progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.label_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 创建主窗口的横向布局，并添加左侧和右侧的layout
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.addWidget(self.left_layout_widget)
        self.main_layout.addWidget(self.right_layout_widget)

        # 为左侧和右侧的layout添加边框
        # self.central_widget.setStyleSheet("border: 1px solid black;")
        # self.left_layout_widget.setStyleSheet("border: 1px solid black;")
        # self.right_layout_widget.setStyleSheet("border: 1px solid black;")
        # 创建模型实例
        self.model = StyleTransferModel(self)

    def resizeEvent(self, event):
        # 在窗口大小改变时，重新设置左侧layout的宽度
        self.left_layout_widget.setFixedWidth(int(self.width() * 0.25))
        super(App, self).resizeEvent(event)

    def openFileNameDialog(self):
        # 清空标签中的图片
        self.progress.setValue(0)
        self.label_input.clear()
        self.label_output.clear()
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Picture", "",
                                                  "All Files (*);;Image Files (*.jpg *.png)", options=options)
        if fileName:
            # 显示原始图片
            pixmap = QPixmap(fileName)
            scaled_pixmap = pixmap.scaled(self.label_input.size(), Qt.KeepAspectRatio)
            self.label_input.setPixmap(scaled_pixmap)
            # 转换图片
            self.progress.setEnabled(True)
            output_image_path = Output_Root + time.strftime('%m%d%H%M%S_', time.localtime(time.time())) +'output.jpg'
            self.model.style_transfer(fileName, output_image_path)
            # 显示输出图片
            pixmap = QPixmap(output_image_path)
            scaled_pixmap = pixmap.scaled(self.label_output.size(), Qt.KeepAspectRatio)
            self.label_output.setPixmap(scaled_pixmap)
            # 显示原始图片
            # pixmap = QPixmap(fileName)
            # scaled_pixmap = pixmap.scaled(self.label_input.size(), Qt.KeepAspectRatio)
            # self.label_input.setPixmap(scaled_pixmap)
            # # 转换图片
            # output_image_path = Output_Root + time.strftime('%m%d%H%M%S_', time.localtime(time.time())) + 'output.jpg'
            # self.thread = StyleTransferThread(self.model, fileName, output_image_path)
            # self.thread.progress_signal.connect(self.setProgress)
            # self.thread.start()

    def setProgress(self, value):
        self.progress.setValue(value)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())