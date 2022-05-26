import numpy as np


def rgb_to_gray(img):
    coeffs = np.array([0.2125, 0.7154, 0.0721])
    return img @ coeffs


def image_quantization(img):
    new_img = np.array(img, copy=True)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                if img[i, j][k] < 32:
                    gray = 0
                elif img[i, j][k] < 64:
                    gray = 32
                elif img[i, j][k] < 96:
                    gray = 64
                elif img[i, j][k] < 128:
                    gray = 96
                elif img[i, j][k] < 160:
                    gray = 128
                elif img[i, j][k] < 192:
                    gray = 160
                elif img[i, j][k] < 224:
                    gray = 192
                else:
                    gray = 224
                new_img[i, j][k] = np.uint8(gray)
    return new_img