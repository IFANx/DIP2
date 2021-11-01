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

# 读取图片
img = cv.imread("D:\\pictures\\car.jpg")

# 添加椒盐噪声，噪声比例为 0.01
img_pepper = sp_noise(img, prob=0.01)

# 显示添加椒盐噪声后的图片
cv.imshow("添加椒盐噪声图片", img_pepper)

# 重复对图像rgb三通道做高斯滤波
b, g, r = cv.split(img_pepper)
img_blue = cv.imshow("Blue_gray", b)
img_green = cv.imshow("Green_grey", g)
img_red = cv.imshow("Red_grey", r)

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