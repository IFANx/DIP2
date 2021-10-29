import cv2 as cv
import numpy as np
# 读取图片
img = cv.imread("D:\\pictures\\car.jpg")

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

# 添加高斯噪声,均值为0，方差为0.001
img_gasuss = gasuss_noise(img, mean=0, var=0.001)

#显示添加高斯噪声后的图片
cv.imshow("添加高斯噪声图片", img_gasuss)


# 调用自行分离rgb三通道函数,对添加完高斯噪声后的图片进行高斯滤波
b, g, r = cv.split(img_gasuss)

# blue通道做标准差σ=1.0的高斯滤波
Gaussian_blue = cv.GaussianBlur(b, (3, 3), 1)
cv.imshow("Gaussian_blue", Gaussian_blue)

# green通道做标准差σ=1.0的高斯滤波
Gaussian_green = cv.GaussianBlur(g, (3, 3), 1)
cv.imshow("Gaussian_green", Gaussian_green)

# red通道做标准差σ=1.0的高斯滤波
Gaussian_red = cv.GaussianBlur(r, (3, 3), 1)
cv.imshow("Gaussian_red", Gaussian_red)

cv.waitKey(0)
cv.destroyAllWindows()