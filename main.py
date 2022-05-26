import cv2
import imutils as imutils
import numpy as np
from numpy import uint8
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.transform import probabilistic_hough_line
from skimage.draw import line
import suauvol
import util
from canny_detector import detect_edges
from morphology import dilate, erode
import time


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


left = imread('resources/left.jpg')
right = imread('resources/right2.jpg')
height = left.shape[0]
width = left.shape[1]

# at least 5% of the image needs to be covered
thresh = height * width * 0.05 * 0.05

# convert image to grayscale
start = time.time()
left_gray = util.rgb_to_gray(left)
end = time.time()
print("RGB to grayscale left.jpg time: ", end - start, "s.")

start = time.time()
right_gray = util.rgb_to_gray(right)
end = time.time()
print("RGB to grayscale right.jpg time: ", end - start, "s.")

imsave("out/left_grayscale.jpg", left_gray)
imsave("out/right_grayscale.jpg", right_gray)

# binarization
start = time.time()
left_bin = suauvol.binarization(left_gray, 0.2, 13, 128)
end = time.time()
print("Sauvol binarization time left.jpg time: ", end - start, "s.")

start = time.time()
right_bin = suauvol.binarization(right_gray, 0.2, 13, 128)
end = time.time()
print("Sauvol binarization time right2.jpg time: ", end - start, "s.")

imsave("out/left_bin.jpg", left_bin)
imsave("out/right_bin.jpg", right_bin)

# edge detection
start = time.time()
left_edges = edge_detection(left_bin)
end = time.time()
print("Edge detection left.jpg time: ", end - start, "s.")

start = time.time()
right_edges = edge_detection(right_bin)
end = time.time()
print("Edge detection right2.jpg time: ", end - start, "s.")

imsave("out/left_edges.jpg", left_edges)
imsave("out/right_edges.jpg", right_edges)

start = time.time()
left_dilate = dilate(left_edges)
end = time.time()
print("Dilate for left.jpg time: ", end - start, "s.")

start = time.time()
right_dilate = dilate(right_edges)
end = time.time()
print("Dilate for right2.jpg time: ", end - start, "s.")

imsave("out/left_dilate.jpg", left_dilate)
imsave("out/right_dilate.jpg", right_dilate)

start = time.time()
left_erode = erode(left_dilate)
end = time.time()
print("Erode left.jpg time: ", end - start, "s.")

start = time.time()
right_erode = erode(right_dilate)
end = time.time()
print("Erode right2.png time: ", end - start, "s.")

imsave("out/left_erode.jpg", left_erode)
imsave("out/right_erode.jpg", right_erode)

start = time.time()
left_lines = line_detection(left_erode)
end = time.time()
print("Line detection left.jpg time: ", end - start, "s. ")


start = time.time()
right_lines = line_detection(right_erode)
end = time.time()
print("Line detection right2.jpg time: ", end - start, "s.")

imsave("out/left_lines.jpg", left_lines)
imsave("out/right_lines.jpg", right_lines)
# diff = compare_images(left_edges, right_edges, method='diff')
start = time.time()
diff = np.abs(left_lines - right_lines)
end = time.time()
print("Diff left and right lines time: ", end - start, "s.")

rectangles = cv2.findContours(img_as_ubyte(diff), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
rectangles = imutils.grab_contours(rectangles)
for c in rectangles:
    (x, y, w, h) = cv2.boundingRect(c)
    if w * h >= thresh:
        left_q = util.image_quantization(left[y:y + h, x:x + w])
        right_q = util.image_quantization(right[y:y + h, x:x + w])
        mean_diff = np.mean(left_q) - np.mean(right_q)
        if abs(mean_diff) > 10:
            cv2.rectangle(right, (x, y), (x + w, y + h), (0, 255, 0), 2)
imsave("out/diff.jpg", right)
