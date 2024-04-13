import torch
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

Model_Path = './trained/pretrained_model.pth'
Image_Root = './images/'
Cache_Root = './cache/'
Output_Root = './output/'

class StyleTransferModel:
    def __init__(self, app):
        # 保存App类的实例
        self.app = app
        # 加载.pth模型文件
        self.model = torch.load(Model_Path)
        self.model.eval()
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'style_transfer', pretrained=True)
        # self.model.eval()

    def Process_image(self,img):
        # 定义转换函数，将输入图片转换为模型所需的格式
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # 加载输入图片
        input_image = img.resize((256, 256)).convert('RGB')
        input_tensor = transform(input_image)
        # 添加batch维度
        input_batch = input_tensor.unsqueeze(0)
        # 运行模型
        with torch.no_grad():
            output =self.model(input_batch)
        #print('Processing')
        # 后处理步骤
        output_image = output[0].cpu().numpy()
        output_image = np.transpose(output_image, (1, 2, 0))
        # 将输出从[-1,1]映射到[0,1]
        output_image = (output_image + 1) / 2.0
        output_image = (output_image * 255).astype(np.uint8)
        output_image = Image.fromarray(output_image, 'RGB')
        return output_image
    def blend_images_Columns(self,image_L, image_R, overlap):
        # OpenCV默认使用BGR色彩空间，所以我们需要从RGB转换为BGR
        image_L = cv2.cvtColor(np.array(image_L), cv2.COLOR_RGB2BGR)
        image_R = cv2.cvtColor(np.array(image_R), cv2.COLOR_RGB2BGR)
        # 定义重叠区域的宽度
        # overlap = 128
        # 分割图片
        left = image_L[:, :image_L.shape[1] - overlap]
        right = image_R[:, overlap:]
        # 创建混合区域
        blend1 = image_L[:, image_L.shape[1] - overlap:]
        blend2 = image_R[:, :overlap]
        # 计算加权平均值
        blended = np.zeros_like(blend1)
        for i in range(overlap):
            alpha = i / overlap
            blended[:, i] = cv2.addWeighted(blend1[:, i], 1 - alpha, blend2[:, i], alpha, 0)
        # 拼接图片
        result = np.hstack((left, blended, right))
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def blend_images_Rows(self,image_T, image_B, overlap):
        # OpenCV默认使用BGR色彩空间，所以我们需要从RGB转换为BGR
        image_T = cv2.cvtColor(np.array(image_T), cv2.COLOR_RGB2BGR)
        image_B = cv2.cvtColor(np.array(image_B), cv2.COLOR_RGB2BGR)
        # 定义重叠区域的高度
        # overlap = 128
        # 分割图片
        top = image_T[:image_T.shape[0] - overlap, :]
        bottom = image_B[overlap:, :]
        # 创建混合区域
        blend1 = image_T[image_T.shape[0] - overlap:, :]
        blend2 = image_B[:overlap, :]
        # 计算加权平均值
        blended = np.zeros_like(blend1)
        for i in range(overlap):
            alpha = i / overlap
            blended[i, :] = cv2.addWeighted(blend1[i, :], 1 - alpha, blend2[i, :], alpha, 0)
        # 拼接图片
        result = np.vstack((top, blended, bottom))
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def Adaptive_SlidingWindow(self,image, window_size, overlap):
        # 获取图片的宽度和高度
        width, height = image.size
        # final_img = Image.new('RGB', (width, height))
        base_step_size = int(window_size * (1 - overlap))  # 计算基本步长，使得窗口之间有重叠
        step_size_x = (width - window_size) // ((width - window_size) // base_step_size)  # 计算实际步长，使得窗口始终充满图像内容
        step_size_y = (height - window_size) // ((height - window_size) // base_step_size)
        totle_img_count = (width//step_size_x)*(height//step_size_y)
        line_img = []
        full_img = []
        # 滑动窗口
        for y in range(0, height - window_size + 1, step_size_y):
            for x in range(0, width - window_size + 1, step_size_x):
                # 切割图片
                window = image.crop((x, y, x + window_size, y + window_size))
                prosedded_img = self.Process_image(window)
                # prosedded_img.save(Cache_Root + str(x) + str(y) + '_img.jpg')
                if len(line_img) == 0:
                    line_img.append(prosedded_img)
                else:
                    blended_img = self.blend_images_Columns(line_img.pop(), prosedded_img,int(((window_size - step_size_x) / window_size) * 256))
                    line_img.append(blended_img)
                if x + step_size_x > width - window_size:
                    # img = line_img.pop()
                    # img.save(Cache_Root + str(y) + '_line_img.jpg')
                    if len(full_img) == 0:
                        full_img.append(line_img.pop())
                    else:
                        # full_img.append(line_img.pop())
                        blended_img = self.blend_images_Rows(full_img.pop(), line_img.pop(),int(((window_size - step_size_y) / window_size) * 256))
                        full_img.append(blended_img)
                # final_img.paste(prosedded_img, (x, y))
        # for i in range(0, len(full_img)):
        # full_img[i].save(Cache_Root + str(i) + '_line_img.jpg')
        return full_img.pop()

    def style_transfer(self, input_image_path, output_image_path):
        # 读取图片
        input_image = Image.open(input_image_path)
        # 更新进度条的值
        self.app.setProgress(50)
        # 使用你的模型进行风格转换
        output_image = self.Adaptive_SlidingWindow(input_image, 512, 0.5)
        # # 这里只是一个示例，你需要替换成自己的代码
        # output_image = input_image
        # 更新进度条的值
        self.app.setProgress(100)
        # 保存输出图片
        output_image.save(output_image_path)