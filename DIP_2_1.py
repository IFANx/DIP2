import cv2 as cv

# 读取图片
img = cv.imread("car.jpg")
cv.imshow("原图", img)

# 调用分离rgb三通道函数
b, g, r = cv.split(img)

# blue通道做标准差σ=1.0的高斯滤波
Gaussian_blue = cv.GaussianBlur(b, (3, 3), 1)

# green通道做标准差σ=1.0的高斯滤波
Gaussian_green = cv.GaussianBlur(g, (3, 3), 1)

# red通道做标准差σ=1.0的高斯滤波
Gaussian_red = cv.GaussianBlur(r, (3, 3), 1)

re = cv.merge([Gaussian_blue, Gaussian_green, Gaussian_red])
cv.imshow("rr", re)

cv.waitKey(0)
cv.destroyAllWindows()
