import math

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, draw, io
from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)
from main import binarization, edge_detection, line_detection
from skimage.io import imread, imsave
from skimage.util import compare_images


def generate_image_test1():
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # draw line
    rr, cc = line(120, 123, 20, 400)
    img[rr, cc, 0] = 255

    # fill polygon
    poly = np.array((
        (300, 300),
        (480, 320),
        (380, 430),
        (220, 590),
        (300, 300),
    ))
    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
    img[rr, cc, 1] = 255

    # fill circle
    rr, cc = disk((200, 200), 100, shape=img.shape)
    img[rr, cc, :] = (255, 255, 0)

    # fill ellipse
    rr, cc = ellipse(300, 300, 100, 200, img.shape)
    img[rr, cc, 2] = 255

    # circle
    rr, cc = circle_perimeter(120, 400, 15)
    img[rr, cc, :] = (255, 0, 0)

    # Bezier curve
    rr, cc = bezier_curve(70, 100, 10, 10, 150, 100, 1)
    img[rr, cc, :] = (255, 0, 0)

    # ellipses
    rr, cc = ellipse_perimeter(120, 400, 60, 20, orientation=math.pi / 4.)
    img[rr, cc, :] = (255, 0, 255)
    rr, cc = ellipse_perimeter(120, 400, 60, 20, orientation=-math.pi / 4.)
    img[rr, cc, :] = (0, 0, 255)
    rr, cc = ellipse_perimeter(120, 400, 60, 20, orientation=math.pi / 2.)
    img[rr, cc, :] = (255,255,255)
    return img


def generate_image_test2():
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # draw line
    rr, cc = line(120, 123, 20, 400)
    img[rr, cc, 0] = 255

    # Bezier curve
    rr, cc = bezier_curve(70, 100, 10, 10, 150, 100, 1)
    img[rr, cc, :] = (255, 0, 0)

    # ellipses
    rr, cc = ellipse_perimeter(120, 400, 60, 20, orientation=math.pi / 4.)
    img[rr, cc, :] = (255, 0, 255)
    rr, cc = ellipse_perimeter(120, 400, 60, 20, orientation=-math.pi / 4.)
    img[rr, cc, :] = (0, 0, 255)
    rr, cc = ellipse_perimeter(120, 400, 60, 20, orientation=math.pi / 2.)
    img[rr, cc, :] = (255,255,255)
    return img


left = generate_image_test1()
right = generate_image_test2()

imsave('right.png', right)
imsave('left.png', left)

left_bin = binarization(left)
right_bin = binarization(right)

left_edges = edge_detection(left_bin)
right_edges = edge_detection(right_bin)

left_lines = line_detection(left_edges)
right_lines = line_detection(right_edges)


cmp1 = compare_images(left_edges, right_edges, method='diff')
cmp2 = compare_images(left_lines, right_lines, method='diff')

imsave("compare_edges.jpg", cmp1)
imsave("compare_lines.jpg", cmp2)