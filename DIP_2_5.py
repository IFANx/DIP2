import math

import cv2 as cv
import numpy as np
import random

# rgb图像转hsi图像
def rgbtohsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv.split(rgb_lwpImg)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            temp=num/den
            if temp>1:
               temp=1
            elif temp<-1:
               temp=-1
            theta = float(np.arccos(temp))
            if den == 0:
                    H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2*3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3*min_RGB/sum

            H = H/(2*3.14159265)
            I = sum/3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H*255
            hsi_lwpImg[i, j, 1] = S*255
            hsi_lwpImg[i, j, 2] = I*255
    return hsi_lwpImg


# hsi图片转rgb图片
def hsitorgb(hsi_img):
    h = int(hsi_img.shape[0])
    w = int(hsi_img.shape[1])
    H, S, I = cv.split(hsi_img)
    H = H / 255.0
    S = S / 255.0
    I = I / 255.0
    bgr_img = hsi_img.copy()
    B, G, R = cv.split(bgr_img)
    for i in range(h):
        for j in range(w):
            if S[i, j] < 1e-6:
                R = I[i, j]
                G = I[i, j]
                B = I[i, j]
            else:
                H[i, j] *= 360
                if H[i, j] > 0 and H[i, j] <= 120:
                    B = I[i, j] * (1 - S[i, j])
                    R = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    G = 3 * I[i, j] - (R + B)
                elif H[i, j] > 120 and H[i, j] <= 240:
                    H[i, j] = H[i, j] - 120
                    R = I[i, j] * (1 - S[i, j])
                    G = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    B = 3 * I[i, j] - (R + G)
                elif H[i, j] > 240 and H[i, j] <= 360:
                    H[i, j] = H[i, j] - 240
                    G = I[i, j] * (1 - S[i, j])
                    B = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    R = 3 * I[i, j] - (G + B)
            bgr_img[i, j, 0] = B * 255
            bgr_img[i, j, 1] = G * 255
            bgr_img[i, j, 2] = R * 255
    return bgr_img


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

# 读取图像
rgb_img = cv.imread("car.jpg")
# 将rgb图像转化为hsi图像
hsi_img = rgbtohsi(rgb_img)

# rgb原图
cv.imshow('rgb_img', rgb_img)

# hsi图片
cv.imshow('hsi_img', hsi_img)

# I为HSI图像的I分量
img = rgbtohsi(rgb_img)
h, s, i = cv.split(img)

# 1.采用高斯滤波器，在I通道做标准差σ=1.0的 高斯滤波
Gaussian_I = cv.GaussianBlur(i, (3, 3), 1)
img1 = cv.merge([h, s, Gaussian_I])
img2 = hsitorgb(img1)
cv.imshow("tyt", img2)

# 2.使用(1)中创建的高斯掩模在RGB图像的各个平面的频域上执行高斯滤波
fi = np.fft.fft2(i)
fishift = np.fft.fftshift(fi)
mask = generate_gaussian_lpf_mask(fishift,64)
fishift_blur = fishift*mask

# 傅里叶逆变换
f1ishift = np.fft.ifftshift(fishift_blur)
ri = np.fft.ifft2(f1ishift)

ri_back = ri.astype(np.uint8)

re = cv.merge([h, s, ri_back])
res = hsitorgb(re)
cv.imshow("gaussi_blur", res)



# 3.添加椒盐噪声，噪声比例为 0.01,并在这些图像上重复(1)和(2)
img_I_pepper = sp_noise(img, prob=0.01)
h, s, i = cv.split(img_I_pepper)
pepper_I = cv.GaussianBlur(i, (3, 3), 1)
img3 = cv.merge([h, s, pepper_I])
img4 = hsitorgb(img3)
cv.imshow("gaussi_pepper_img ", img4)

# 4. 应用中值滤波，使用5x5邻域，对这些噪声损坏的图像(高斯噪声损坏和“椒盐”噪声损坏的RGB图像)




key = cv.waitKey(0)
cv.destroyAllWindows()


