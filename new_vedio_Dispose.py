import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import cv2
import time
from screeninfo import get_monitors

Model_Path = './trained/pretrained_model.pth'
Image_Root = './images/'
Cache_Root = './cache/'
Output_Root = './output/'

def Process_image(img,model):
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
        output = model(input_batch)
    print('Processing')
    # 后处理步骤
    output_image = output[0].cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))
    # 将输出从[-1,1]映射到[0,1]
    output_image = (output_image + 1) / 2.0
    output_image = (output_image * 255).astype(np.uint8)
    output_image = Image.fromarray(output_image, 'RGB')
    return output_image

def Adaptive_SlidingWindow(image, window_size,overlap):
    # 获取图片的宽度和高度
    width, height = image.size
    #final_img = Image.new('RGB', (width, height))
    base_step_size = int(window_size * (1 - overlap))# 计算基本步长，使得窗口之间有重叠
    step_size_x = (width - window_size) // ((width - window_size) // base_step_size)# 计算实际步长，使得窗口始终充满图像内容
    step_size_y = (height - window_size) // ((height - window_size) // base_step_size)
    line_img = []
    full_img = []
    # 滑动窗口
    for y in range(0, height - window_size + 1, step_size_y):
        for x in range(0, width - window_size + 1, step_size_x):
            # 切割图片
            window = image.crop((x, y, x + window_size, y + window_size))
            prosedded_img = Process_image(window, model)
            # prosedded_img.save(Cache_Root + str(x) + str(y) + '_img.jpg')
            if len(line_img) == 0:
                line_img.append(prosedded_img)
            else:
                blended_img = blend_images_Columns(line_img.pop(), prosedded_img, int(((window_size-step_size_x)/window_size)*256))
                line_img.append(blended_img)
            if x + step_size_x > width - window_size :
                if len(full_img) == 0:
                    full_img.append(line_img.pop())
                else:
                    blended_img = blend_images_Rows(full_img.pop(), line_img.pop(), int(((window_size-step_size_y)/window_size) * 256))
                    full_img.append(blended_img)
    return full_img.pop()

def blend_images_Columns(image_L, image_R, overlap):
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
def blend_images_Rows(image_T, image_B, overlap):
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



if __name__ == '__main__':
    # 加载.pth模型文件
    model = torch.load(Model_Path)
    model.eval()
    # #加载图片文件
    # Image_Name = '_DSC1917.jpg'
    # Image_Path = Image_Root + Image_Name
    # input_image = Image.open(Image_Path)
    # # 获取图像的宽度和高度
    # width, height = input_image.size
    # # 对图片进行处理
    # final_img = Adaptive_SlidingWindow(input_image, 512, 0.5)
    # width, height = final_img.size
    # # 保存新图像
    # reslotion = str(width) + 'x' + str(height)
    # file_name = Output_Root + time.strftime('%m%d%H%M%S_', time.localtime(time.time())) + reslotion + '.jpg'
    # final_img.save(file_name)
    # 读取视频
    cap = cv2.VideoCapture('./vedios/mount_1920x1080.mp4')
    # 获取原视频的宽度，高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 创建一个VideoWriter对象，用于输出视频
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8))
            # 对每一帧进行处理
            processed_frame = Process_image(frame,model)
            processed_frame = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
            # 将处理后的帧放大到原视频尺寸
            processed_frame = cv2.resize(processed_frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            # 将处理后的帧写入新的视频
            out.write(processed_frame)
        else:
            break

    # 释放资源
    cap.release()
    out.release()




