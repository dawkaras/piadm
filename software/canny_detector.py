import numpy as np


def mask_generation(t, sigma):
    temp = -np.log(t) * 2 * (sigma ** 2)
    half = np.round(np.sqrt(temp))
    y, x = np.meshgrid(range(-int(half), int(half) + 1), range(-int(half), int(half) + 1))
    return x, y


def gaussian(x, y, sigma):
    temp = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
    return np.exp(-temp)


def pad(img, kernel):
    r, c = img.shape
    kr, kc = kernel.shape
    padded = np.zeros((r + kr, c + kc), dtype=img.dtype)
    insert = np.uint(kr / 2)
    padded[int(insert): int(insert + r), int(insert): int(insert + c)] = img
    return padded


def smooth(img, kernel=None):
    if kernel is None:
        mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    else:
        mask = kernel
    i, j = mask.shape
    output = np.zeros((img.shape[0], img.shape[1]))
    image_padded = pad(img, mask)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            output[x, y] = (mask * image_padded[x:x + i, y:y + j]).sum() / mask.sum()
    return output


def f(fx, fy, sigma):
    return (fx ** 2 + fy ** 2) / (2 * sigma ** 2)


def calculate_mask(fx, fy, sigma):
    temp = f(fx, fy, sigma)
    gx = -((fy * np.exp(-temp)) / sigma ** 2)
    gx = (gx * 255)
    gy = -((fx * np.exp(-temp)) / sigma ** 2)
    gy = (gy * 255)
    return gx, gy


def apply_mask(image, kernel):
    i, j = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)
    image_padded = pad(image, kernel)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            output[x, y] = (kernel * image_padded[x:x + i, y:y + j]).sum()
    return output


def gradient_magnitude(fx, fy):
    mag = np.sqrt((fx ** 2) + (fy ** 2))
    mag = mag * 100 / mag.max()
    return np.around(mag)


def gradient_direction(fx, fy):
    g_dir = np.rad2deg(np.arctan2(fy, fx)) + 180
    return g_dir


def digitize_angle(angle):
    quantized = np.zeros((angle.shape[0], angle.shape[1]))
    for i in range(angle.shape[0]):
        for j in range(angle.shape[1]):
            if 0 <= angle[i, j] <= 22.5 or 157.5 <= angle[i, j] <= 202.5 or 337.5 < angle[i, j] < 360:
                quantized[i, j] = 0
            elif 22.5 <= angle[i, j] <= 67.5 or 202.5 <= angle[i, j] <= 247.5:
                quantized[i, j] = 1
            elif 67.5 <= angle[i, j] <= 122.5 or 247.5 <= angle[i, j] <= 292.5:
                quantized[i, j] = 2
            elif 112.5 <= angle[i, j] <= 157.5 or 292.5 <= angle[i, j] <= 337.5:
                quantized[i, j] = 3
    return quantized


def non_max_supp(qn, mag, d):
    m = np.zeros(qn.shape)
    a, b = np.shape(qn)
    for i in range(a - 1):
        for j in range(b - 1):
            if qn[i, j] == 0:
                if mag[i, j - 1] < mag[i, j] or mag[i, j] > mag[i, j + 1]:
                    m[i, j] = d[i, j]
                else:
                    m[i, j] = 0
            if qn[i, j] == 1:
                if mag[i - 1, j + 1] <= mag[i, j] or mag[i, j] >= mag[i + 1, j - 1]:
                    m[i, j] = d[i, j]
                else:
                    m[i, j] = 0
            if qn[i, j] == 2:
                if mag[i - 1, j] <= mag[i, j] or mag[i, j] >= mag[i + 1, j]:
                    m[i, j] = d[i, j]
                else:
                    m[i, j] = 0
            if qn[i, j] == 3:
                if mag[i - 1, j - 1] <= mag[i, j] or mag[i, j] >= mag[i + 1, j + 1]:
                    m[i, j] = d[i, j]
                else:
                    m[i, j] = 0
    return m


def double_thresholding(g_suppressed, low_threshold, high_threshold):
    threshold = np.zeros(g_suppressed.shape)
    for i in range(0, g_suppressed.shape[0]):  # loop over pixels
        for j in range(0, g_suppressed.shape[1]):
            if g_suppressed[i, j] < low_threshold:  # lower than low threshold
                threshold[i, j] = 0
            elif low_threshold <= g_suppressed[i, j] < high_threshold:  # between thresholds
                threshold[i, j] = 128
            else:  # higher than high threshold
                threshold[i, j] = 255
    return threshold


def hysteresis(threshold):
    g_strong = np.zeros(threshold.shape)
    for i in range(0, threshold.shape[0]):  # loop over pixels
        for j in range(0, threshold.shape[1]):
            val = threshold[i, j]
            if val == 128:  # check if weak edge connected to strong
                if threshold[i - 1, j] == 255 or threshold[i + 1, j] == 255 or threshold[
                    i - 1, j - 1] == 255 or threshold[i + 1, j - 1] == 255 or threshold[i - 1, j + 1] == 255 or \
                        threshold[i + 1, j + 1] == 255 or threshold[i, j - 1] == 255 or threshold[i, j + 1] == 255:
                    g_strong[i, j] = 255  # replace weak edge as strong
            elif val == 255:
                g_strong[i, j] = 255  # strong edge remains as strong edge
    return g_strong


def detect_edges(image, sigma, t=0.1):
    x, y = mask_generation(t, sigma)
    gauss = gaussian(x, y, sigma)
    gx, gy = calculate_mask(x, y, sigma)
    smooth_img = smooth(image, gauss)
    fx = apply_mask(smooth_img, gx)
    fy = apply_mask(smooth_img, gy)
    mag = gradient_magnitude(fx, fy)
    mag = mag.astype(int)
    angle = gradient_direction(fx, fy)
    quantized = digitize_angle(angle)
    nms = non_max_supp(quantized, angle, mag)
    threshold = double_thresholding(nms, 1, 70)
    return hysteresis(threshold)
