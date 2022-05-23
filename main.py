import cv2
import imutils as imutils
import numpy as np
from math import sqrt
from numpy import uint8
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_sauvola
from skimage.transform import probabilistic_hough_line
from skimage import feature
import time
from skimage.draw import line
from skimage.util import compare_images


def integral_image(image, height, width):
    I = np.zeros((height, width))
    II = np.zeros((height, width))
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
                bw[h][w] = 0 if image[h][w] < avg * (1 + k * (sqrt(sumSqr / s - avg * avg) / R - 1)) else 255
    return bw


def binarization(img):
    grayscale = rgb2gray(img)
    thresh_sauvola = threshold_sauvola(grayscale, window_size=25)
    return grayscale > thresh_sauvola


def edge_detection(grayscale_img):
    return feature.canny(grayscale_img, sigma=0.5)


def line_detection(img):
    lines = probabilistic_hough_line(img, threshold=10, line_length=10, line_gap=5)
    lines_img = np.zeros((img.shape[0], img.shape[1]), dtype=uint8)

    for hline in lines:
        p0, p1 = hline
        rr, cc = line(p0[1], p0[0], p1[1], p1[0])
        # drawing red lines on original image
        lines_img[rr, cc] = 255
    return lines_img


def image_quantization(img):
    new_img = np.array(img, copy = True)
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


# image = imread('test.jpg')
# height = image.shape[0]
# width = image.shape[1]
#
# # binarization
# start = time.time()
# binarized = binarization(image)
# end = time.time()
# imsave('binarized.jpg', img_as_ubyte(binarized))
# print("Binarization took ", end - start, "ms")
#
# # edge detection with canny
# start = time.time()
# edges = edge_detection(binarized)
# end = time.time()
# imsave('edges.jpg', img_as_ubyte(edges))
# print("Canny edge detection took ", end - start, "ms")
#
# ## line detectinon with hough
# start = time.time()
# lines = line_detection(edges)
# end = time.time()
# print('Probabilistic hough lines detection took ', end - start, 'ms')
# imsave('lines.jpg', lines)
left = imread('left.jpg')
right = imread('right2.jpg')
height = left.shape[0]
width = left.shape[1]

# at least 5% of the image needs to be covered
thresh = height * width * 0.05 * 0.05

left_bin = binarization(left)
right_bin = binarization(right)

left_edges = edge_detection(left_bin)
right_edges = edge_detection(right_bin)
#
# left_lines = line_detection(left_edges)
# right_lines = line_detection(right_edges)
#
#
cmp1 = compare_images(left_edges, right_edges, method='diff')
# cmp2 = compare_images(left_lines, right_lines, method='diff')

# diff = left.copy()
# cv2.absdiff(left, right, diff)
# gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# for i in range(0, 3):
#     dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
# (T, thresh) = cv2.threshold(dilated, 3, 255, cv2.THRESH_BINARY)
# print(thresh)
cnts = cv2.findContours(img_as_ubyte(cmp1), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if w * h >= thresh:
        left_q = image_quantization(left[y:y+h, x:x+w])
        right_q = image_quantization(right[y:y+h, x:x+w])
        mean_diff = np.mean(left_q) - np.mean(right_q)
        if abs(mean_diff) > 10:
            cv2.rectangle(right, (x, y), (x + w, y + h), (0, 255, 0), 2)

imsave("diff.jpg", right)
