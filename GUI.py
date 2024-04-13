



# class StyleTransferThread(QThread):
#     progressChanged = pyqtSignal(int)
#     def __init__(self, model, input_image_path, output_image_path):
#         super().__init__()
#         self.model = model
#         self.input_image_path = input_image_path
#         self.output_image_path = output_image_path
#     def run(self):
#         self.model.style_transfer(self.input_image_path, self.output_image_path, self.setProgress)
#     def setProgress(self, value):
#         self.progressChanged.emit(value)

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QProgressBar, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from Model import StyleTransferModel
import time

Model_Path = './trained/pretrained_model.pth'
Image_Root = './images/'
Cache_Root = './cache/'
Output_Root = './output/'

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口初始大小
        self.resize(800, 600)
        # 创建垂直布局
        layout = QVBoxLayout()
        self.setLayout(layout)
        # 创建"选择图片"按钮
        btn = QPushButton('Select Picture', self)
        btn.clicked.connect(self.openFileNameDialog)
        layout.addWidget(btn)
        # 创建进度条
        self.progress = QProgressBar(self)
        layout.addWidget(self.progress)
        # 创建水平布局用于显示图片
        hlayout = QHBoxLayout()
        layout.addLayout(hlayout)
        # 创建标签用于显示原始图片
        self.label_input = QLabel(self)
        self.label_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hlayout.addWidget(self.label_input)
        # 创建标签用于显示输出图片
        self.label_output = QLabel(self)
        self.label_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hlayout.addWidget(self.label_output)
        # 创建模型实例
        self.model = StyleTransferModel(self)

    def openFileNameDialog(self):
        # 清空标签中的图片
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
            output_image_path = Output_Root + time.strftime('%m%d%H%M%S_', time.localtime(time.time())) +'output.jpg'
            self.model.style_transfer(fileName, output_image_path)
            # 显示输出图片
            pixmap = QPixmap(output_image_path)
            scaled_pixmap = pixmap.scaled(self.label_output.size(), Qt.KeepAspectRatio)
            self.label_output.setPixmap(scaled_pixmap)

    def setProgress(self, value):
        self.progress.setValue(value)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())