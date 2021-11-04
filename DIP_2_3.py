import cv2 as cv
import random
import numpy as np

# 添加椒盐噪声函数, prob:噪声比例
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

# 生成高斯卷积核
def generate_gaussian_lpf_mask(shifted_fft, radius) -> np.ndarray:
    """
    Generate a gauss LPF mask for frequency domain filtering.
    :param shifted_fft: the fft spectrum to filter
    :param radius: the lpf filter's radius
    :return: the mask
    """
    size_x = shifted_fft.shape[0]
    size_y = shifted_fft.shape[1]
    mask = np.zeros((size_x, size_y))
    x0 = np.floor(size_x / 2)
    y0 = np.floor(size_y / 2)
    for i in range(size_x):
        for j in range(size_y):
            d = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
            mask[i][j] = np.exp((-1) * d ** 2 / 2 / (radius ** 2))
    return mask

# 读取图片
img = cv.imread("car.jpg")

#保存图片
# cv.imwrite("images\\3_original_image.jpg", img)

# 添加椒盐噪声，噪声比例为 0.01
img_pepper = sp_noise(img, prob=0.01)

# 显示添加椒盐噪声后的图片
cv.imshow("img_pepper", img_pepper)
# 保存图片
# cv.imwrite("images\\3_pepper_image.jpg", img_pepper)


# 重复对图像rgb三通道做高斯滤波
# 分离添加完椒盐噪声的图像的三通道
b, g, r = cv.split(img_pepper)

# blue通道做标准差σ=1.0的高斯滤波
Gaussian_blue = cv.GaussianBlur(b, (3, 3), 1)

# green通道做标准差σ=1.0的高斯滤波
Gaussian_green = cv.GaussianBlur(g, (3, 3), 1)

# red通道做标准差σ=1.0的高斯滤波
Gaussian_red = cv.GaussianBlur(r, (3, 3), 1)

# 合并三通道
Gaussian_filter_pepper_img1 = cv.merge([Gaussian_blue, Gaussian_green, Gaussian_red])

#显示高斯滤波处理完的添加椒盐噪声的图片
cv.imshow("Gaussian_filter_pepper_img1", Gaussian_filter_pepper_img1)

#保存图片
# cv.imwrite("images\\3_Gaussian_filter_pepper_img1.jpg", Gaussian_filter_pepper_img1)


# 在RGB图像的各个平面的频域上执行高斯滤波
b, g, r = cv.split(img_pepper)
# 傅里叶变换
fb = np.fft.fft2(b)
fg = np.fft.fft2(g)
fr = np.fft.fft2(r)

fbshift = np.fft.fftshift(fb)
fgshift = np.fft.fftshift(fg)
frshift = np.fft.fftshift(fr)

mask = generate_gaussian_lpf_mask(fbshift,64)
fbshift_blur = fbshift*mask
fgshift_blur = fgshift*mask
frshift_blur = frshift*mask

# 傅里叶逆变换
f1bshift = np.fft.ifftshift(fbshift_blur)
f1gshift = np.fft.ifftshift(fgshift_blur)
f1rshift = np.fft.ifftshift(frshift_blur)

rb = np.fft.ifft2(f1bshift)
rg = np.fft.ifft2(f1gshift)
rr = np.fft.ifft2(f1rshift)

rb_back = rb.astype(np.uint8)
rg_back = rg.astype(np.uint8)
rr_back = rr.astype(np.uint8)

Gaussian_filter_pepper_img2 = cv.merge([rb_back, rg_back, rr_back])

#显示高斯滤波处理完的添加椒盐噪声的图片
cv.imshow("Gaussian_filter_pepper_img2", Gaussian_filter_pepper_img2)

#保存图片
# cv.imwrite("images\\3_Gaussian_filter_pepper_img2.jpg", Gaussian_filter_pepper_img2)

cv.waitKey(0)
cv.destroyAllWindows()