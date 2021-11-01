import cv2 as cv
import random
import numpy as np

# 添加椒盐噪声function(), prob:噪声比例
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

# 添加高斯掩膜函数,mean : 均值,var : 方差
def gasuss_noise(image, mean=0, var=0.001):

    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    output = image + noise
    if output.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    output = np.clip(output, low_clip, 1.0)
    output = np.uint8(output*255)
    return output


# 读取图片
img = cv.imread("D:\\pictures\\car.jpg")

# 添加椒盐噪声，噪声比例为 0.01
img_pepper = sp_noise(img, prob=0.01)

# 添加高斯噪声,均值为0，方差为0.001
img_gasuss = gasuss_noise(img, mean=0, var=0.001)

# 中值滤波 处理添加椒盐噪声的图片
img_median_pepper = cv.medianBlur(img_pepper, 5)

# 中值滤波 处理添加高斯噪声的图片
img_median_gasuss = cv.medianBlur(img_gasuss, 5)

# 原图
cv.imshow("原图", img)

# 添加椒盐噪声后图像
cv.imshow("添加椒盐噪声后图像", img_pepper)

# 添加高斯噪声后图像
cv.imshow("添加高斯噪声后图像", img_gasuss)

# 中值滤波处理椒盐噪声后图像
cv.imshow("中值滤波处理椒盐噪声后图像", img_median_pepper)

# 中值滤波处理高斯噪声后图像
cv.imshow("中值滤波处理高斯噪声后图像", img_median_gasuss)

cv.waitKey(0)
cv.destroyAllWindows()