import os
import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QProgressBar, QSizePolicy, QLineEdit, QMainWindow
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QIntValidator, QDoubleValidator, QImage
from Model import StyleTransferModel
import time

Model_Path = './trained/pretrained_model.pth'
Image_Root = './images/'
Cache_Root = './cache/'
Output_Root = './output/'

# class StyleTransferThread(QThread):
#     progress_signal = pyqtSignal(int)
#     def __init__(self, model, input_image_path, output_image_path):
#         super().__init__()
#         self.model = model
#         self.input_image_path = input_image_path
#         self.output_image_path = output_image_path
#     def run(self):
#         self.model.style_transfer(self.input_image_path, self.output_image_path, self.update_progress)
#     def update_progress(self, progress):
#         self.progress_signal.emit(progress)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):

        self.setWindowTitle('基于对抗学习的宫崎骏动漫风格图像变换方法实现')
        self.setMinimumSize(750, 600)
        self.setMaximumSize(750, 600)
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
        self.open_image_button = QPushButton('打开图像')
        self.batch_openImg_button = QPushButton('打开文件夹')
        self.batch_openImg_button.clicked.connect(self.openAllFilesInFolder)
        self.open_image_button.clicked.connect(self.openSignalFile)
        self.open_video_button = QPushButton('打开视频')
        self.open_video_button.clicked.connect(self.open_video)
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.parameter1_input = QLineEdit('512')
        self.validatorInt = QIntValidator(128, 640, self)# 创建一个 QIntValidator 对象，设置范围为 1 到 100
        self.parameter1_input.setValidator(self.validatorInt)
        self.parameter2_input = QLineEdit('0.5')
        self.validatorDou = QDoubleValidator(0.0, 1.0, 1, self)  # 创建一个 QDouValidator 对象，设置范围为 1 到 100
        self.parameter2_input.setValidator(self.validatorDou)
        self.input_layout = QHBoxLayout()
        self.input_layout.addWidget(self.parameter1_input)
        self.input_layout.addWidget(self.parameter2_input)
        self.left_layout.addWidget(self.open_image_button)
        self.left_layout.addWidget(self.batch_openImg_button)
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

        self.cap = None
        self.out = None
        self.total_frames = 0
        self.current_frame = 0

        # 为左侧和右侧的layout添加边框
        # self.central_widget.setStyleSheet("border: 1px solid black;")
        # self.left_layout_widget.setStyleSheet("border: 1px solid black;")
        # self.right_layout_widget.setStyleSheet("border: 1px solid black;")

    def resizeEvent(self, event):
        # 在窗口大小改变时，重新设置左侧layout的宽度
        self.left_layout_widget.setFixedWidth(int(self.width() * 0.25))
        super(App, self).resizeEvent(event)

    def get_input(self):
        # 从 QLineEdit 对象获取用户输入
        parameter1_input = self.parameter1_input.text()
        parameter2_input = self.parameter2_input.text()
        return int(parameter1_input), float(parameter2_input)

    def openSignalFile(self):
        win_size,overlap = self.get_input()
        # 创建模型实例
        self.model = StyleTransferModel(self, win_size, overlap)
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

    def openAllFilesInFolder(self):
        win_size, overlap = self.get_input()
        # 创建模型实例
        self.model = StyleTransferModel(self, win_size, overlap)
        # 清空标签中的图片
        self.progress.setValue(0)
        self.label_input.clear()
        self.label_output.clear()
        options = QFileDialog.Options()
        dirName = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dirName:
            files = os.listdir(dirName)
            images = [f for f in files if f.endswith(('.jpg', '.png'))]  # 只处理jpg和png文件
            for image in images:
                fileName = os.path.join(dirName, image)
                # 显示原始图片
                pixmap = QPixmap(fileName)
                scaled_pixmap = pixmap.scaled(self.label_input.size(), Qt.KeepAspectRatio)
                self.label_input.setPixmap(scaled_pixmap)
                # 转换图片
                self.progress.setEnabled(True)
                output_image_path = Output_Root + time.strftime('%m%d%H%M%S_',
                                                                time.localtime(time.time())) + '_output.jpg'
                self.model.style_transfer(fileName, output_image_path)
                # 显示输出图片
                self.label_output.clear()
                pixmap = QPixmap(output_image_path)
                scaled_pixmap = pixmap.scaled(self.label_output.size(), Qt.KeepAspectRatio)
                self.label_output.setPixmap(scaled_pixmap)
                # 这里可能需要一些延时，以便你能看到每个图片的处理过程
    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Video')
        if filename:
            self.cap = cv2.VideoCapture(filename)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            #self.progress.setMaximum(self.total_frames)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                       (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer.start(42)
    def process_frame(self, frame):
        # 对图像进行处理，返回处理后的图像
        self.model = StyleTransferModel(self, 256, 0.5)
        return self.model.Process_image(frame)

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if ret:
            self.current_frame += 1
            self.progress.setValue(int((self.current_frame/float(self.total_frames)*100)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame_width, frame_height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            scaled_pixmap = pix.scaled(self.label_input.size(), Qt.KeepAspectRatio)
            self.label_input.setPixmap(scaled_pixmap)
            frame = Image.fromarray(frame.astype(np.uint8))
            # 对每一帧进行处理
            processed_frame = self.process_frame(frame)
            processed_frame = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
            # 将处理后的帧放大到原视频尺寸
            processed_frame = cv2.resize(processed_frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            # 将处理后的帧写入新的视频
            self.out.write(processed_frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = QImage(processed_frame, frame_width, frame_height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            scaled_pixmap = pix.scaled(self.label_output.size(), Qt.KeepAspectRatio)
            self.label_output.setPixmap(scaled_pixmap)


            if self.current_frame >= self.total_frames:
                self.cap.release()
                self.out.release()
                self.timer.stop()
                self.progress.setValue(0)
                os.startfile(os.getcwd())
    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()
    def setProgress(self, value):
        self.progress.setValue(value)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())