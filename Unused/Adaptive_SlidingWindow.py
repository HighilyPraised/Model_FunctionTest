from PIL import Image


Image_Root = './images/'
Cache_Root = './cache/'
Output_Root = './output/'
def sliding_window(image, window_size,overlap):#本程序用于动态滑动窗口测试
    # 获取图片的宽度和高度
    width, height = image.size
    # 计算基本步长，使得窗口之间有重叠
    base_step_size = int(window_size * (1 - overlap))
    # 计算实际步长，使得窗口始终充满图像内容
    step_size_x = (width - window_size) // ((width - window_size) // base_step_size)
    step_size_y = (height - window_size) // ((height - window_size) // base_step_size)
    # 滑动窗口
    for y in range(0, height - window_size + 1, step_size_y):
        for x in range(0, width - window_size + 1, step_size_x):
            # 切割图片
            window = image.crop((x, y, x + window_size, y + window_size))

            window.convert('RGB').save(Cache_Root +str(x)+str(y) +'_existing_img.jpg')

image = Image.open(Image_Root+'6oxgp6_3840x2160.jpg')
sliding_window(image, 1024, 0.5)