import math

import numpy as np


def rgb_to_gray(img):
    coeffs = np.array([0.2125, 0.7154, 0.0721])
    return img @ coeffs


def integral_image(image, height, width):
    i = np.zeros((height, width))
    ii = np.zeros((height, width))
    for h in range(0, height):
        sum_i = 0
        sum_sqr = 0
        for w in range(0, width):
            sum_i += image[h][w]
            sum_sqr += image[h][w] * image[h][w]
            if h == 0:
                i[h][w] = sum_i
                ii[h][w] = sum_sqr
            else:
                i[h][w] = i[h - 1][w] + sum_i
                ii[h][w] = ii[h - 1][w] + sum_sqr
    return i, ii


def sauvola(image, I, II, height, width, k, r, R):
    bw = np.zeros((height, width))
    s = (2 * r + 1) * (2 * r + 1)
    for h in range(0, height):
        for w in range(0, width):
            if r + 1 <= h < height - r - 1 and r + 1 <= w < width - r - 1:
                sum = I[h + r][w + r] + I[h - r - 1][w - r - 1] - I[h - r - 1][w + r] - \
                      I[h + r][w - r - 1]
                sumSqr = II[h + r][w + r] + II[h - r - 1][w - r - 1] - II[h - r - 1][w + r] - \
                         II[h + r][w - r - 1]
                avg = sum / s
                bw[h][w] = 0 if image[h][w] < avg * (1 + k * (math.sqrt(abs(sumSqr / s - avg * avg)) / R - 1)) else 255
    return bw


def binarization(image, k, r, R):
    I, II = integral_image(image, image.shape[0], image.shape[1])
    return sauvola(image, I, II, image.shape[0], image.shape[1], k, r, R)
