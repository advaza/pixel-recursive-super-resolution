# Copyright (c) 2017 Lightricks. All rights reserved.

import tensorflow as tf
import numpy as np
import os, cv2

from skimage.transform import resize, rescale, AffineTransform
from skimage.util import random_noise
from skimage.io import imread, imsave
from skimage.restoration import denoise_bilateral
from PIL import ImageFilter, Image

# RGB channel image
IMAGE_N_DIM = 3
BORDER_SIZE = 32

interpolation_order = {
    'Nearest-neighbor': 0,
    'Bi-linear': 1,
    'Bi-quadratic': 2,
    'Bi-cubic': 3,
    'Bi-quartic': 4,
    'Bi-quintic': 5,
}


def affine_transform(image, output_shape):

    rows, cols = output_shape[0], output_shape[1]
    orig_rows, orig_cols = image.shape[0], image.shape[1]

    row_scale = float(orig_rows) / rows
    col_scale = float(orig_cols) / cols
    if rows == 1 and cols == 1:
        tform = AffineTransform(translation=(orig_cols / 2.0 - 0.5,
                                             orig_rows / 2.0 - 0.5))
    else:
        # 3 control points necessary to estimate exact AffineTransform
        src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
        dst_corners = np.zeros(src_corners.shape, dtype=np.double)
        # take into account that 0th pixel is at position (0.5, 0.5)
        dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
        dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

        tform = AffineTransform()
        tform.estimate(src_corners, dst_corners)

    return tform

def to_ind(array):
    return np.round(array).astype(dtype=np.int64)


def image_dwnscale(image, scale=1):

    height, width, channels = image.shape
    rows = to_ind(scale * np.arange(height * (1./scale)))
    cols = to_ind(scale * np.arange(width * (1./scale)))
    row_inds = np.repeat(rows, len(cols))
    col_inds = np.tile(cols, len(rows))
    rescaled = image[row_inds, col_inds, :].reshape(len(rows), len(cols), channels)

    return rescaled


def bilateral_rescale(image, scale=2, win_size=None, sigma_color=None, sigma_spatial=1,
                      bins=10000, mode='constant'):

    rescaled = image_dwnscale(image, scale)
    return denoise_bilateral(
        image=rescaled,
        win_size=win_size,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        bins=bins,
        mode=mode,
        multichannel=True,
    )


def ndarray_to_pil(image):
    return Image.fromarray(np.uint8(image * 255.0), mode="RGB")


def pil_to_ndarray(pil_image):
    image = np.array(pil_image)
    image = np.float32(image)
    image /= 255.
    return image


def unsharp_mask(image, radius=2, percent=150, threshold=3):

    pil_image = ndarray_to_pil(image)
    unsharp_mask_filter = ImageFilter.UnsharpMask(
        radius=radius,
        percent=percent,
        threshold=threshold
    )

    pil_result = pil_image.filter(unsharp_mask_filter)
    return pil_to_ndarray(pil_result)


def guided_filter(image, guide=None, radius=2, eps=0.01):

    if guide is None:
        guide = image

    image = image.astype(np.float32)
    guide = guide.astype(np.float32)
    return cv2.ximgproc.guidedFilter(guide=guide, src=image, radius=radius, eps=eps)


def calc_scale(src, target):
    scale = target.shape[0] / src.shape[0]
    return scale


def im_read(image_file):
    image = imread(image_file)
    if image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    return image


def as_batch(image):
    if len(image.shape) == 3:
        return tf.expand_dims(image, 0)
    return image


def load_image(image_path):
    """
    Read an image from image path, convert its dtyte to float and divide by 255.
    :param image_path:
    :return:
    """
    image = im_read(image_path)
    image = np.ndarray.astype(image, dtype=np.float32) / 255.0
    return image


def save_image(image, image_path):
    image = image * 255.0
    image = np.uint8(image)
    imsave(image_path, image)


def _smallest_size_at_least_op(height, width, smallest_size):

    smallest_size = tf.cast(smallest_size, dtype=tf.float32)
    height = tf.cast(height, dtype=tf.float32)
    width = tf.cast(width, dtype=tf.float32)

    # Set scale using Tensors to implements:
    # scale = smallest_size / width if height > width else smallest_size / height
    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_size / width,
                    lambda: smallest_size / height)

    new_height = height * scale
    new_width = width * scale

    return tf.cast(new_height, dtype=tf.int32), tf.cast(new_width, dtype=tf.int32)


def aspect_preserving_resize_SKImage(image, smallest_size, get_new_shape=False,
                                     interp_order='Bi-linear'):

    height, width, channels = image.shape
    new_height, new_width = _smallest_size_at_least(height, width, smallest_size=smallest_size)

    resized_image = resize(
        image=image,
        output_shape=[new_height, new_width],
        order=interpolation_order[interp_order],
        mode='constant',
    )

    if get_new_shape:
        return resized_image, [new_height, new_width, channels]

    return resized_image


def add_noise(image, noise_type, amount=0.05, var=0.01):
    noise_description = noise_type

    if noise_type in ['gaussian', 'speckle']:
        noise_description += '_var_' + str(var)
        noised_image = random_noise(image, mode=noise_type, var=var)

    elif noise_type in ['pepper', 's&p']:
        noise_description += '_amount_' + str(amount)
        noised_image = random_noise(image, mode=noise_type, amount=amount)

    else:
        noised_image = random_noise(image, mode=noise_type)

    return noised_image, noise_description


def aspect_preserving_resize_TF(image, smallest_size, get_new_shape=False):
    """
    
    :param image:
    :param smallest_size:
    :return:
    """
    smallest_size_tensor = tf.constant(smallest_size, dtype=tf.int32, shape=[])
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least_op(
        height,
        width,
        smallest_size=smallest_size_tensor)

    resized_image = tf.image.resize_bilinear(
        as_batch(image),
        [new_height, new_width],
        align_corners=False
    )
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])

    if get_new_shape:
        height, width, channels = image.get_shape().as_list()
        new_height, new_width = _smallest_size_at_least(height, width, smallest_size=smallest_size)
        return resized_image, [new_height, new_width, channels]

    return resized_image


def add_reflect_border(image, border_size=BORDER_SIZE):
    return tf.pad(
        image,
        [[0, 0], [border_size, border_size], [border_size, border_size], [0, 0]],
        mode='REFLECT'
    )


def remove_border(image, border_size=BORDER_SIZE):
    return image[:, border_size:-border_size, border_size:-border_size, :]


def _smallest_size_at_least(height, width, smallest_size):
    height = np.float32(height)
    width = np.float32(width)
    smallest_size = np.float32(smallest_size)
    scale = smallest_size/width if height > width else smallest_size/height
    new_height = height * scale
    new_width = width * scale
    return np.int32(new_height), np.int32(new_width)


def initialize_dir_if_not_exists(dir_name):
    """
    Create the directory if it is not exists already.
    :param dir_name:
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
