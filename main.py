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

from canny_detector import detect_edges
from morphology import dilate, erode


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
    # res = feature.canny(grayscale_img, sigma=0.5)
    res = detect_edges(grayscale_img, 0.5)
    return res


def line_detection(img):
    lines = probabilistic_hough_line(img, threshold=10, line_length=10, line_gap=5)
    # lines = detect_lines(img)
    lines_img = np.zeros((img.shape[0], img.shape[1]), dtype=uint8)
    for hline in lines:
        p0, p1 = hline
        rr, cc = line(p0[1], p0[0], p1[1], p1[0])
        #     # drawing red lines on original image
        lines_img[rr, cc] = 255
    return lines_img


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
imsave("left_edges.jpg", left_edges)
imsave("right_edges.jpg", right_edges)
left_dilate = dilate(left_edges)
right_dilate = dilate(right_edges)
imsave("left_dilate.jpg", left_dilate)
imsave("right_dilate.jpg", right_dilate)
left_erode = erode(left_dilate)
right_erode = erode(right_dilate)
imsave("left_erode.jpg", left_erode)
imsave("right_erode.jpg", right_erode)
left_lines = line_detection(left_erode)
right_lines = line_detection(right_erode)
imsave("left_lines.jpg", left_lines)
imsave("right_lines.jpg", right_lines)
# diff = compare_images(left_edges, right_edges, method='diff')
diff = compare_images(left_lines, right_lines, method='diff')

rectangles = cv2.findContours(img_as_ubyte(diff), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
rectangles = imutils.grab_contours(rectangles)
for c in rectangles:
    (x, y, w, h) = cv2.boundingRect(c)
    if w * h >= thresh:
        left_q = image_quantization(left[y:y+h, x:x+w])
        right_q = image_quantization(right[y:y+h, x:x+w])
        mean_diff = np.mean(left_q) - np.mean(right_q)
        if abs(mean_diff) > 10:
            cv2.rectangle(right, (x, y), (x + w, y + h), (0, 255, 0), 2)
imsave("diff.jpg", right)
