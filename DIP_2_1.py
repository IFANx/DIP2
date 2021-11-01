import cv2 as cv
import numpy as np
# 读取图片
img = cv.imread("D:\\pictures\\car.jpg")

# 调用自行分离rgb三通道函数
b, g, r = cv.split(img)
img_blue = cv.imshow("Blue_gray", b)
img_green = cv.imshow("Green_grey", g)
img_red = cv.imshow("Red_grey", r)

zeros = np.zeros(img.shape[:2], dtype = "uint8")
merged_r = cv.merge([zeros,zeros,r])

zeros = np.zeros(img.shape[:2], dtype = "uint8")
merged_g = cv.merge([zeros,g,zeros])

zeros = np.zeros(img.shape[:2], dtype = "uint8")
merged_b = cv.merge([b,zeros,zeros])

# blue通道做标准差σ=1.0的高斯滤波
Gaussian_blue = cv.GaussianBlur(merged_b, (3, 3), 1)
cv.imshow("Gaussian_blue", Gaussian_blue)

# green通道做标准差σ=1.0的高斯滤波
Gaussian_green = cv.GaussianBlur(merged_g, (3, 3), 1)
cv.imshow("Gaussian_green", Gaussian_green)

# red通道做标准差σ=1.0的高斯滤波
Gaussian_red = cv.GaussianBlur(merged_r, (3, 3), 1)
cv.imshow("Gaussian_red", Gaussian_red)


cv.waitKey(0)
cv.destroyAllWindows()
