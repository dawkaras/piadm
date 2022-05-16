import numpy as np
from math import sqrt
from numpy import uint8
from skimage.io import imread, imsave
from skimage.color import rgb2gray


def integral_image(image, height, width):
    I = [[0 for y in range(width)] for x in range(height)]
    II = [[0 for y in range(width)] for x in range(height)]
    for h in range(0, height):
        sum = 0
        sumSqr = 0
        for w in range(0, width):
            sum += image[h][w]
            sumSqr += image[h][w] * image[h][w]
            if h == 0:
                I[h][w] = sum
                II[h][w] = sumSqr
            else:
                I[h][w] = I[h - 1][w] + sum
                II[h][w] = II[h - 1][w] + sumSqr
    return I, II


def sauvola(image, I, II, height, width, k, r, R):
    bw = [[0 for y in range(width)] for x in range(height)]
    s = (2 * r + 1) * (2 * r + 1)
    for h in range(0, height):
        for w in range(0, width):
            if r + 1 <= h < height - r - 1 and r + 1 <= w < width - r - 1:
                sum = I[h + r][w + r] + I[h - r - 1][w - r - 1] - I[h - r - 1][w + r] - \
                      I[h + r][w - r - 1]
                sumSqr = II[h + r][w + r] + II[h - r - 1][w - r - 1] - II[h - r - 1][w + r] - \
                         II[h + r][w - r - 1]
                avg = sum / s
                bw[h][w] = 0 if image[h][w] < avg * (1 + k * (sqrt(sumSqr / s - avg * avg) / R - 1)) else 255
    return bw


image = imread('test.jpg')
height = image.shape[0]
width = image.shape[1]
grayscale = rgb2gray(image)
I, II = integral_image(grayscale, height, width)
out = sauvola(grayscale, I, II, height, width, 0.1, 13, 128)
imsave('test2.jpg', np.array(out, dtype=uint8))
