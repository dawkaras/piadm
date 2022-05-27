import cv2
import imutils as imutils
import numpy as np
from cv2 import dilate, erode
from numpy import uint8
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters.thresholding import threshold_sauvola
from skimage.io import imread, imsave
from skimage.transform import probabilistic_hough_line
from skimage.draw import line
from skimage.util import compare_images

import util
import time


def sauvol(img):
    thresh_sauvola = threshold_sauvola(img, window_size=25)
    return img > thresh_sauvola


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


left = imread('resources/frame1.jpg')
right = imread('resources/frame2.jpg')
height = left.shape[0]
width = left.shape[1]

# at least 1% of the image needs to be covered
thresh = height * width * 0.01 * 0.01

# convert image to grayscale
start = time.time()
left_gray = rgb2gray(left)
end = time.time()
print("RGB to grayscale left.jpg time: ", end - start, "s.")

start = time.time()
right_gray = rgb2gray(right)
end = time.time()
print("RGB to grayscale right.jpg time: ", end - start, "s.")

imsave("obrazy_i_dane/left_grayscale.jpg", img_as_ubyte(left_gray))
imsave("obrazy_i_dane/right_grayscale.jpg", img_as_ubyte(right_gray))

# binarization
start = time.time()
left_bin = sauvol(left_gray)
end = time.time()
print("Sauvol binarization time left.jpg time: ", end - start, "s.")

start = time.time()
right_bin = sauvol(right_gray)
end = time.time()
print("Sauvol binarization time right2.jpg time: ", end - start, "s.")

imsave("obrazy_i_dane/left_bin.jpg", img_as_ubyte(left_bin))
imsave("obrazy_i_dane/right_bin.jpg", img_as_ubyte(right_bin))

# edge detection
start = time.time()
left_edges = img_as_ubyte(canny(left_bin, sigma=0.5))
end = time.time()
print("Edge detection left.jpg time: ", end - start, "s.")

start = time.time()
right_edges = img_as_ubyte(canny(right_bin, sigma=0.5))
end = time.time()
print("Edge detection right.jpg time: ", end - start, "s.")

imsave("obrazy_i_dane/left_edges.jpg", left_edges)
imsave("obrazy_i_dane/right_edges.jpg", right_edges)
start = time.time()
left_dilate = dilate(left_edges, None, iterations=1)
end = time.time()
print("Dilate for left.jpg time: ", end - start, "s.")

start = time.time()
right_dilate = dilate(right_edges, None, iterations=1)
end = time.time()
print("Dilate for right.jpg time: ", end - start, "s.")

imsave("obrazy_i_dane/left_dilate.jpg", left_dilate)
imsave("obrazy_i_dane/right_dilate.jpg", right_dilate)

start = time.time()
left_erode = erode(left_dilate, None, iterations=1)
end = time.time()
print("Erode left.jpg time: ", end - start, "s.")

start = time.time()
right_erode = erode(right_dilate, None, iterations=1)
end = time.time()
print("Erode right.jpg time: ", end - start, "s.")

imsave("obrazy_i_dane/left_erode.jpg", left_erode)
imsave("obrazy_i_dane/right_erode.jpg", right_erode)

start = time.time()
left_lines = line_detection(left_erode)
end = time.time()
print("Line detection left.jpg time: ", end - start, "s. ")

start = time.time()
right_lines = line_detection(right_erode)
end = time.time()
print("Line detection right.jpg time: ", end - start, "s.")

imsave("obrazy_i_dane/left_lines.jpg", left_lines.astype(np.uint8))
imsave("obrazy_i_dane/right_lines.jpg", right_lines.astype(np.uint8))

start = time.time()
diff = compare_images(left_lines, right_lines, method='diff')
end = time.time()
print("Diff left and right lines time: ", end - start, "s.")

rectangles = cv2.findContours(img_as_ubyte(diff), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rectangles = imutils.grab_contours(rectangles)
for c in rectangles:
    (x, y, w, h) = cv2.boundingRect(c)
    if w * h >= thresh:
        left_q = util.image_quantization(left[y:y + h, x:x + w])
        right_q = util.image_quantization(right[y:y + h, x:x + w])
        mean_diff = np.mean(left_q) - np.mean(right_q)
        if abs(mean_diff) > 10:
            cv2.rectangle(right, (x, y), (x + w, y + h), (0, 255, 0), 2)
imsave("obrazy_i_dane/diff.jpg", right)
