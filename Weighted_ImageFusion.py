
import cv2
import numpy as np

def smooth_seam(image):
    # 将图像分割成四个128*128的图像
    img1 = image[0:128, 0:128]
    img2 = image[0:128, 128:256]
    img3 = image[128:256, 0:128]
    img4 = image[128:256, 128:256]

    # 对每两个相邻的图像在接缝处进行加权平均
    img12 = cv2.addWeighted(img1[:, -10:], 0.5, img2[:, :10], 0.5, 0)
    img34 = cv2.addWeighted(img3[:, -10:], 0.5, img4[:, :10], 0.5, 0)

    # 将处理过的图像重新拼接
    img1[:, -10:] = img12
    img2[:, :10] = img12
    img3[:, -10:] = img34
    img4[:, :10] = img34

    top = np.concatenate((img1, img2), axis=1)
    bottom = np.concatenate((img3, img4), axis=1)

    # 拼接上下两部分
    result = np.concatenate((top, bottom), axis=0)

    return result

def multiband_blending(img1, img2, overlap):
    # 将图像转换为YUV颜色空间
    yuv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    yuv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)
    # 分别对Y、U、V通道进行多频段混合
    for i in range(3):
        mb1 = cv2.pyrMeanShiftFiltering(yuv1[:, :, i], 21, 51)
        mb2 = cv2.pyrMeanShiftFiltering(yuv2[:, :, i], 21, 51)
        yuv1[:, :, i] = cv2.addWeighted(mb1, 0.5, yuv2[:, :, i], 0.5, 0)
        yuv2[:, :, i] = cv2.addWeighted(mb2, 0.5, yuv1[:, :, i], 0.5, 0)
    # 将图像从YUV颜色空间转换回BGR颜色空间
    result1 = cv2.cvtColor(yuv1, cv2.COLOR_YUV2BGR)
    result2 = cv2.cvtColor(yuv2, cv2.COLOR_YUV2BGR)
    # 返回混合后的图像
    return result1, result2


# 读取两张RGB图像
img1 = cv2.imread('./lpls/384128_existing_img.jpg')
img2 = cv2.imread('./lpls/384128_window_img.jpg')
# 确保两张图片的尺寸相同
assert img1.shape == img2.shape, "Images must have the same shape."
# 获取图像的尺寸
height, width, _ = img1.shape
# 创建权重矩阵
# weights = np.zeros((height, width))
# # # 左半部分的权重从1变化到0
# # weights[:, :width // 2] = np.linspace(1, 0, width // 2)
# # # 右半部分的权重从0变化到1
# # weights[:, width // 2:] = np.linspace(0, 1, width - width // 2)
#
# # 整个权重矩阵从右到左变化，从1到0
# weights[:] = np.linspace(1, 0, width)
#
# # 将权重矩阵扩展到与图像相同的形状
# weights = np.repeat(weights[:, :, np.newaxis], 3, axis=2)
# # 使用权重矩阵进行图像融合
# output = weights * img1 + (1 - weights) * img2
# # 将输出转换为uint8，因为imshow函数需要的输入类型是uint8
# output = output.astype(np.uint8)
# # 显示融合后的图片
# cv2.imshow('Blended', output)
# cv2.waitKey(0)

mask = (img1 == 0)
# 从image2中选择像素，并复制到image1的对应位置
cv2.imshow('existing_black', img1)
img1[mask]=img2[mask]
cv2.imshow('existing', img1)
cv2.imshow('window', img2)
# 创建权重矩阵
weights1 = np.zeros((height, width))
weights2 = np.zeros((height, width))
# img1的权重矩阵从左到右变化，从1到0
weights1[:] = np.linspace(1, 0, width)
# img2的权重矩阵从右到左变化，从1到0
weights2[:] = np.linspace(0, 1, width)
# 将权重矩阵扩展到与图像相同的形状
weights1 = np.repeat(weights1[:, :, np.newaxis], 3, axis=2)
weights2 = np.repeat(weights2[:, :, np.newaxis], 3, axis=2)
# 使用权重矩阵进行图像融合
output1 = weights1 * img1 + weights2 * img2
cv2.imshow('Output1', output1.astype(np.uint8))
# 创建权重矩阵
weights1 = np.zeros((height, width))
weights2 = np.zeros((height, width))
# img1的权重矩阵从上到下变化，从1到0
weights1 = np.linspace(1, 0, height)[:, np.newaxis]
# img2的权重矩阵从下到上变化，从1到0
weights2 = np.linspace(0, 1, height)[:, np.newaxis]
# 将权重矩阵扩展到与图像相同的形状
weights1 = np.repeat(weights1[:, :, np.newaxis], 3, axis=2)
weights2 = np.repeat(weights2[:, :, np.newaxis], 3, axis=2)
# 使用权重矩阵进行图像融合
output = weights1 * output1 + weights2 * img2

# # 处理图像
# result = smooth_seam(output)

# 读取图像
image = output
# 将图像分割成四个128*128的图像
img1 = image[0:128, 0:128]
img2 = image[0:128, 128:256]
img3 = image[128:256, 0:128]
img4 = image[128:256, 128:256]
# 对每两个相邻的图像进行多频段混合
img1, img2 = multiband_blending(img1, img2, 10)
img3, img4 = multiband_blending(img3, img4, 10)
# 将处理过的图像重新拼接
top = np.concatenate((img1, img2), axis=1)
bottom = np.concatenate((img3, img4), axis=1)
# 拼接上下两部分
result = np.concatenate((top, bottom), axis=0)

cv2.imshow('Blended', output.astype(np.uint8))
# 显示图像
cv2.imshow('Result', result.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()


