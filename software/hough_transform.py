import cv2
import numpy as np
from numpy import uint8


def line_detection_vectorized(edge_image, num_rhos=180, num_thetas=180, t_count=220):
    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
    #
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    d_theta = 180 / num_thetas
    d_rho = (2 * d) / num_rhos
    #
    thetas = np.arange(0, 180, step=d_theta)
    rhos = np.arange(-d, d, step=d_rho)
    #
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    #
    edge_points = np.argwhere(edge_image != 0)
    edge_points = edge_points - np.array([[edge_height_half, edge_width_half]])
    #
    rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))
    #
    accumulator, theta_val, rho_val = np.histogram2d(
        np.tile(thetas, rho_values.shape[0]),
        rho_values.ravel(),
        bins=[thetas, rhos]
    )
    accumulator = np.transpose(accumulator)
    lines = np.argwhere(accumulator > t_count)
    res = np.zeros((edge_image.shape[0], edge_image.shape[1]), dtype=uint8)
    for line in lines:
        y, x = line
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return res


def detect_lines(image):
    return line_detection_vectorized(image)

