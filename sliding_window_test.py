import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import cv2
import time

Split_mode = True

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

def SlidingWindow(img,new_width, new_height):
    # 创建一个新的图像用于填充原始图像
    new_img = Image.new('RGB', (new_width, new_height))
    # 将原始图像粘贴到新图像中
    new_img.paste(img)
    new_img.save(Cache_Root + 'temp_img.jpg')

    final_img = Image.new('RGB', (new_width, new_height))
    for y in range(new_height-256,-1, -128):
        for x in range(0, new_width-256  , 128):
            input_image = new_img.crop((x, y, x + 256  , y + 256))
            input_image.save(Cache_Root +str(x)+str(y) + '_crop_img.jpg')
            window_image = Process_image(input_image,model)
            window_image.save(Cache_Root + str(x) + str(y) + '_window_img.jpg')
            if x==0 and y==0:
                final_img.paste(window_image, (x, y))
            else:
                existing_image = final_img.crop((x, y, x + 256, y + 256))
                # existing_image_np = np.array(existing_image)
                # window_image_np = np.array(window_image)
                # black_mask = (existing_image_np ==0 )
                # mask = (existing_image_np !=0)
                # existing_image_np[black_mask] = window_image_np[black_mask]
                # existing_image = Image.fromarray(existing_image_np, 'RGB')
                existing_image.save(Cache_Root +str(x)+str(y) +'_existing_img.jpg')
                blended_window = blend_images(existing_image, window_image, 'Weighted')
                final_img.paste(blended_window, (x, y))
    return final_img


def Adaptive_SlidingWindow(image, window_size,overlap):
    # 获取图片的宽度和高度
    width, height = image.size
    final_img = Image.new('RGB', (width, height))
    base_step_size = int(window_size * (1 - overlap))# 计算基本步长，使得窗口之间有重叠
    step_size_x = (width - window_size) // ((width - window_size) // base_step_size)# 计算实际步长，使得窗口始终充满图像内容
    step_size_y = (height - window_size) // ((height - window_size) // base_step_size)
    # 滑动窗口
    for y in range(0, height - window_size + 1, step_size_y):
        for x in range(0, width - window_size + 1, step_size_x):
            # 切割图片
            window = image.crop((x, y, x + window_size, y + window_size))
            prosedded_img = Process_image(window, model)
            final_img.paste(prosedded_img, (x, y))
            # if x == 0 and y == 0:
            #     final_img.paste(prosedded_img, (x, y))
            # else:
            #     existing_image = final_img.crop((x, y, x + window_size, y + window_size))
            #     existing_image_np = np.array(existing_image)
            #     prosedded_img_np = np.array(prosedded_img)
            #     mask = (existing_image_np == 0)
            #     existing_image_np[mask] = prosedded_img_np[mask]
            #     existing_image = Image.fromarray(existing_image_np, 'RGB')
            #     blended_window = blend_images(existing_image, prosedded_img_np, 'Weighted')
            #     final_img.paste(blended_window, (x, y))
    return final_img

def blend_images(image1, image2, blend_factor):# 定义图像融合函数
    # 将PIL图像转换为numpy数组以进行计算
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    match blend_factor:
        case 'AVG':# 平均融合
            return Image.blend(image1, image2, alpha=0.3)
        case 'Weighted':# 加权融合
            height, width, _ = image1_array.shape
            # 找出image1中的黑色部分
            mask = (image1_array == 0)
            # 从image2中选择像素，并复制到image1的对应位置
            image1_array[mask]=image2_array[mask]
            # 创建权重矩阵
            weights1 = np.zeros((height, width))
            weights2 = np.zeros((height, width))
            weights3 = np.zeros((height, width))
            weights4 = np.zeros((height, width))
            # 初始化权重矩阵
            weights1[:] = np.linspace(1, 0, width)# 从左到右变化，从1到0
            weights2[:] = np.linspace(0, 1, width)# 从右到左变化，从1到0
            weights3 = np.linspace(1, 0, height)[:, np.newaxis]# 从上到下变化，从1到0
            weights4 = np.linspace(0, 1, height)[:, np.newaxis]# 下到上变化，从1到0
            weights1 = np.repeat(weights1[:, :, np.newaxis], 3, axis=2)
            weights2 = np.repeat(weights2[:, :, np.newaxis], 3, axis=2)
            weights3 = np.repeat(weights3[:, :, np.newaxis], 3, axis=2)
            weights4 = np.repeat(weights4[:, :, np.newaxis], 3, axis=2)# 将权重矩阵扩展3通道
            # 使用权重矩阵进行图像融合
            output1 = weights1 * image1_array + weights2 * image2_array
            # cv2.imshow('Output1', output1.astype(np.uint8))
            blended_array = weights3 * output1 + weights4 * image2_array
            # 将融合后的图像转换回PIL图像
            # Image.fromarray(blended_array.astype(np.uint8))
            # Image.fromarray(image2_array.astype(np.uint8))
            return Image.fromarray(blended_array.astype(np.uint8))

if __name__ == '__main__':
    # 加载.pth模型文件
    model = torch.load(Model_Path)
    model.eval()
    #加载图片文件
    #Image_Names = os.listdir(Image_Root)
    Image_Name = 'x8pjml_1920x1080.jpg'
    Image_Path = Image_Root + Image_Name
    input_image = Image.open(Image_Path)
    if Split_mode:
        # 获取图像的宽度和高度
        width, height = input_image.size
        # 计算需要填充的宽度和高度
        # new_width = (width + 255) // 256 * 256
        # new_height = (height + 255) // 256 * 256


        # 对图片进行处理
        final_img = Adaptive_SlidingWindow(input_image, 512, 0.5)
        # 保存新图像
        reslotion = str(width) + 'x' + str(height)
        file_name = Output_Root+time.strftime('%m%d%H%M%S_',time.localtime(time.time()))+reslotion+'.jpg'
        final_img.save(file_name)
        print('Image saved: '+file_name)
    else:
        output_image = Process_image(input_image,model)
        file_name = Output_Root + time.strftime('%m%d%H%M%S_', time.localtime(time.time())) + '256x256.jpg'
        # 使用OpenCV保存图像
        cv2.imwrite(file_name,cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

