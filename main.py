import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
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
    if Split_mode:
        output_image = Image.fromarray(output_image,'RGB')
        return output_image
    else:
        return output_image

def Cut_Image(img,new_width, new_height):
    # 创建一个新的图像用于填充原始图像
    new_img = Image.new('RGB', (new_width, new_height))
    # 将原始图像粘贴到新图像中
    new_img.paste(img)
    new_img.save(Cache_Root + 'temp_img.jpg')
    # 分割图像
    input_images = [new_img.crop((x, y, x + 256, y + 256)) for x in range(0, new_width, 256) for y in range(0, new_height, 256)]
    return input_images


if __name__ == '__main__':
    # 加载.pth模型文件
    model = torch.load(Model_Path)
    model.eval()
    #加载图片文件
    #Image_Names = os.listdir(Image_Root)
    Image_Name = '6oxgp6_1920x1080.jpg'
    Image_Path = Image_Root + Image_Name
    input_image = Image.open(Image_Path)

    if Split_mode:
        # 获取图像的宽度和高度
        width, height = input_image.size
        # 计算需要填充的宽度和高度
        new_width = (width + 255) // 256 * 256
        new_height = (height + 255) // 256 * 256
        # 分割图像
        input_images = Cut_Image(input_image,new_width, new_height)
        # 风格化
        output_images = [Process_image(img,model) for img in input_images]
        # 合并图像
        final_img = Image.new('RGB', ( new_width, new_height))
        k = 0
        for x in range(0, new_width, 256):
            for y in range(0, new_height, 256):
                final_img.paste(output_images[k], (x, y))
                k += 1
        # 保存新图像
        final_img = final_img.crop((0, 0, width, height))
        reslotion = str(width) + 'x' + str(height)
        file_name = Output_Root+time.strftime('%m%d%H%M%S_',time.localtime(time.time()))+reslotion+'.jpg'
        final_img.save(file_name)
        print('Image saved: '+file_name)
    else:
        output_image = Process_image(input_image,model)
        file_name = Output_Root + time.strftime('%m%d%H%M%S_', time.localtime(time.time())) + '256x256.jpg'
        # 使用OpenCV保存图像
        cv2.imwrite(file_name,cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

