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

# 添加高斯噪声函数,mean : 均值,var : 方差
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
img = cv.imread("car.jpg")
# #保存图片
# cv.imwrite("images\\4_original_image.jpg", img)

# 添加椒盐噪声，噪声比例为 0.01
img_pepper = sp_noise(img, prob=0.01)
# 添加椒盐噪声后图像
cv.imshow("add salt_pepper", img_pepper)
# #保存图片
# cv.imwrite("images\\4_pepper_image.jpg", img_pepper)


# 添加高斯噪声,均值为0，方差为0.001
img_gaussian = gasuss_noise(img, mean=0, var=0.001)
# 添加高斯噪声后图像
cv.imshow("add gasuss_noise", img_gaussian)
# #保存图片
# cv.imwrite("images\\4_gaussian_image.jpg", img_gaussian)


# 中值滤波 处理添加椒盐噪声的图片
Median_filter_pepper_img = cv.medianBlur(img_pepper, 5)
# 中值滤波处理椒盐噪声后图像
cv.imshow("Median_filter_pepper_img", Median_filter_pepper_img)
# #保存图片
# cv.imwrite("images\\4_Median_filter_pepper_img.jpg", Median_filter_pepper_img)


# 中值滤波 处理添加高斯噪声的图片
Median_filter_gasussian_img = cv.medianBlur(img_gaussian, 5)
# 中值滤波处理高斯噪声后图像
cv.imshow("Median_filter_gasussian_img", Median_filter_gasussian_img)
# #保存图片
# cv.imwrite("images\\4_Median_filter_gasussian_img.jpg", Median_filter_gasussian_img)


cv.waitKey(0)
cv.destroyAllWindows()