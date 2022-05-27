import numpy as np


def dilate(image, dilation_level=3):
    dilation_level = 3 if dilation_level < 3 else dilation_level
    structuring_kernel = np.full(shape=(dilation_level, dilation_level), fill_value=255)
    orig_shape = image.shape
    pad_width = dilation_level - 2
    image_pad = np.pad(array=image, pad_width=pad_width, mode='constant')
    pad_img_shape = image_pad.shape
    h_reduce, w_reduce = (pad_img_shape[0] - orig_shape[0]), (pad_img_shape[1] - orig_shape[1])
    sub_matrix = np.array([
        image_pad[i:(i + dilation_level), j:(j + dilation_level)]
        for i in range(pad_img_shape[0] - h_reduce) for j in range(pad_img_shape[1] - w_reduce)
    ])
    image_dilate = np.array([255 if (i == structuring_kernel).any() else 0 for i in sub_matrix])
    image_dilate = image_dilate.reshape(orig_shape)

    return image_dilate


def erode(image, erosion_level=3):
    erosion_level = 3 if erosion_level < 3 else erosion_level
    structuring_kernel = np.full(shape=(erosion_level, erosion_level), fill_value=255)
    orig_shape = image.shape
    pad_width = erosion_level - 2
    image_pad = np.pad(array=image, pad_width=pad_width, mode='constant')
    pad_img_shape = image_pad.shape
    h_reduce, w_reduce = (pad_img_shape[0] - orig_shape[0]), (pad_img_shape[1] - orig_shape[1])
    sub_matrix = np.array([
        image_pad[i:(i + erosion_level), j:(j + erosion_level)]
        for i in range(pad_img_shape[0] - h_reduce) for j in range(pad_img_shape[1] - w_reduce)
    ])
    image_erode = np.array([255 if (i == structuring_kernel).all() else 0 for i in sub_matrix])
    image_erode = image_erode.reshape(orig_shape)
    return image_erode
