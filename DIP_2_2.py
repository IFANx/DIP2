import cv2 as cv
import numpy as np

# 创建高斯卷积核函数
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
img = cv.imread('car.jpg') #直接读为灰度图像

# 保存图片
# cv.imwrite("images\\2_original_image.jpg", img)

# 调用分离rgb三通道函数
b, g, r = cv.split(img)

# 每一个通道做傅里叶变换
fb = np.fft.fft2(b)
fg = np.fft.fft2(g)
fr = np.fft.fft2(r)

fbshift = np.fft.fftshift(fb)
fgshift = np.fft.fftshift(fg)
frshift = np.fft.fftshift(fr)

# 产生高斯卷积核
mask = generate_gaussian_lpf_mask(fbshift, 64)
# 对每一个频域做高斯模糊
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

# 处理完的三个频域做merge
Gaussian_filter_img = cv.merge([rb_back, rg_back, rr_back])
cv.imshow("Gaussian_filter_img", Gaussian_filter_img)

# 保存图片
# cv.imwrite("images\\2_Gaussian_filter_img.jpg", Gaussian_filter_img)


cv.waitKey(0)
cv.destroyAllWindows()



