import cv2
import imutils as imutils
import numpy as np
from numpy import uint8
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_sauvola
from skimage.transform import probabilistic_hough_line
from skimage.draw import line
from skimage.util import compare_images
import suauvol
from canny_detector import detect_edges
from morphology import dilate, erode


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


left = imread('resources/left.jpg')
right = imread('resources/right2.jpg')
height = left.shape[0]
width = left.shape[1]

# at least 5% of the image needs to be covered
thresh = height * width * 0.05 * 0.05

# convert image to grayscale
left_gray = suauvol.rgb_to_gray(left)
right_gray = suauvol.rgb_to_gray(right)

# binarization
left_bin = suauvol.binarization(left_gray, 0.2, 13, 128)
right_bin = suauvol.binarization(right_gray, 0.2, 13, 128)
imsave("out/left_bin.jpg", left_bin)
imsave("out/right_bin.jpg", right_bin)

# edge detection
left_edges = edge_detection(left_bin)
right_edges = edge_detection(right_bin)
imsave("out/left_edges.jpg", left_edges)
imsave("out/right_edges.jpg", right_edges)

left_dilate = dilate(left_edges)
right_dilate = dilate(right_edges)
imsave("out/left_dilate.jpg", left_dilate)
imsave("out/right_dilate.jpg", right_dilate)
left_erode = erode(left_dilate)
right_erode = erode(right_dilate)
imsave("out/left_erode.jpg", left_erode)
imsave("out/right_erode.jpg", right_erode)
left_lines = line_detection(left_erode)
right_lines = line_detection(right_erode)
imsave("out/left_lines.jpg", left_lines)
imsave("out/right_lines.jpg", right_lines)
# diff = compare_images(left_edges, right_edges, method='diff')
diff = np.abs(left_lines - right_lines)

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
imsave("out/diff.jpg", right)
