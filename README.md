Model_Path = './trained/pretrained_model.pth'模型路径 <br />
Image_Root = './images/'存放测试图片的路径 <br />
Cache_Root = './cache/'缓存路径 <br />
Output_Root = './output/'模型输出的风格化图像的路径 <br />
请确保这四个路径存在 <br /> <br />

main.py 用于测试模型的基础功能，即输入任意尺寸图片，输出风格化图片，目前功能已完善 <br />
GUI.py pyQt5搭建的图形界面，仍需修改 <br />
Model.py 模型功能文件 <br />
Vedio_Dispose.py 该脚本用于视频风格化测试，输入视频文件输出风格化的视频，系统开销太大，需要修改 <br /> <br />

unused 所有已弃用的脚本 <br />
Adaptive_SlidingWindow.py 用于测试自适应的滑动窗口切片，功能在main已实现，脚本已弃用 <br />
Weighted_ImageFusion.py 用于测试图片的加权融合，功能在main已实现，脚本已弃用 <br />
Vedio_Cut.py 将任意长度的视频切割成指定秒数的短视频，已弃用 <br />