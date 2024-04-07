import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time


Model_Path = './trained/pretrained_model.pth'
Image_Root = './images/'
Cache_Root = './cache/'
Output_Root = './output/'

def Process_image(img,model):

    # 定义转换函数，将输入图片转换为模型所需的格式
    transform = transforms.Compose([
        #transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # 加载输入图片
    input_image = img.resize((256, 256)).convert('RGB')
    input_tensor = transform(input_image)  # 添加batch维度
    input_batch = input_tensor.unsqueeze(0)
    # 运行模型
    with torch.no_grad():
        output = model(input_batch)
    # 后处理步骤

    output_image = output[0].cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))
    # 将输出从[-1,1]映射到[0,1]
    output_image = (output_image + 1) / 2.0
    output_image = (output_image * 255).astype(np.uint8)

    # 使用OpenCV保存图像
    cv2.imwrite('output.jpg', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # 加载.pth模型文件
    model = torch.load(Model_Path)
    model.eval()
    #加载图片文件
    input_image = Image.open(Image_Root+'vme61m_1920x1080.jpg')
    Process_image(input_image,model)
