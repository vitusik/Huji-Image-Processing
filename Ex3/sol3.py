import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import os


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def read_image(filename, representation):
    """
    :param filename: path to the image we want to read
    :param representation: 1 means the image needs to be transformed to grayscale, 0 leaves the image as is
    :return: numpy array in the shape of the image with values between 0 and 1, data type of each element is float64
    """
    base_im = imread(filename)
    # if the input is image in grayscale there is no point in using rgb2gray on it
    if representation == 1 and len(base_im.shape) == 3:
        im = rgb2gray(base_im)
    else:
        im = base_im
        im = im.astype(np.float64)
        im /= 255
    return im


def create_kernel(kernel_size):
    """
    Helper function that creates a 1D Gaussian kernel
    :param kernel_size: len of the desired kernel
    :return: 1D Gaussian kernel
    """
    if kernel_size == 1:
        return np.ones(shape=(1, 1))
    base = np.asarray([1, 1])
    ker = base
    for i in range(2, kernel_size):
        ker = np.convolve(ker, base)
    ker = ker.reshape((kernel_size, 1))
    return ker / np.cumsum(ker)[-1]


def reduce(image, kernel):
    """
    Helper func that reduces the given image size by 2 on each axis
    :param image: 2D array of values in range [0,1] that represent the image
    :param kernel: 1D row Gaussian kernel that used to blur the image
    :return: the input image with its shape divided by 2 on each axis
    """
    blurred_im = convolve(image, kernel, mode='constant')
    blurred_im = convolve(blurred_im, kernel.T, mode='constant')
    reduced_im = np.ndarray((int(image.shape[0] / 2), int(image.shape[1] / 2)))
    reduced_im[:] = blurred_im[::2, ::2]
    return reduced_im


def expand(image, kernel):
    """
    Helper func that expands the given image size by 2 on each axis
    :param image: 2D array of values in range [0,1] that represent the image
    :param kernel: 1D row Gaussian kernel that used to blur the image
    :return: the input image with its shape multiplied by 2 on each axis
    """
    expanded_im = np.zeros((int(image.shape[0] * 2), int(image.shape[1] * 2)))
    expanded_im[::2, ::2] = image
    kernel = kernel * 2
    expanded_im = convolve(expanded_im, kernel, mode='constant')
    expanded_im = convolve(expanded_im, kernel.T, mode='constant')
    return expanded_im


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: 2D array of values in range [0,1] that represent the image
    :param max_levels: amount of pyramids to create, can be less
    :param filter_size: size of the Gaussian filter
    :return: list of all of the Gaussian pyramids from the original to the smallest,
             and row vector of the Gaussian filter
    """
    filter_vec = create_kernel(filter_size)
    pyr = []
    pyr.append(im)
    amount_of_levels = int(np.min([np.floor(np.log2(im.shape[0])), np.floor(np.log2(im.shape[1])), max_levels]))
    for i in range(0, amount_of_levels - 1):
        pyr.append(reduce(pyr[i], filter_vec))
    return pyr, filter_vec.T


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    :param im: 2D array of values in range [0,1] that represent the image
    :param max_levels: amount of pyramids to create, can be less
    :param filter_size: size of the Gaussian filter
    :return: list of all of the Laplacian pyramids from the largest to the smallest,
             and row vector of the Gaussian filter
    """
    gauss, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    amount_of_levels = int(np.min([np.floor(np.log2(im.shape[0])), np.floor(np.log2(im.shape[1])), max_levels]))
    lap = []
    for i in range(len(gauss) - 1):
        lap.append(gauss[i] - expand(gauss[i + 1], filter_vec))
    lap.append(gauss[-1])
    return lap, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: list of Laplacian pyramids of an image
    :param filter_vec: the Gaussian filter used to blur each level
    :param coeff: row vector in the length of lpyr
    :return: the original image
    """
    im = lpyr[-1]
    for i in range(len(lpyr)):
        lpyr[i] = lpyr[i] * coeff[i]
    for j in range(len(lpyr) - 1, 0, -1):
        im = lpyr[j - 1] + expand(im, filter_vec)
    return im


def render_pyramid(pyr, levels):
    '''
    :param pyr: list of Gaussian or Laplacian pyramids of an image
    :param levels: the amount of levels of the pyramids to stack together horizontally
    :return: pyramids from the largest one to the pyr[levels] stacked together horizontally
    '''
    x = pyr[0].shape[0]
    y = 0
    rendered_per = []
    for i in range(levels):
        max_val = np.amax(pyr[i])
        min_val = np.amin(pyr[i])
        pyr[i] = (pyr[i] - min_val) / (max_val - min_val)
        y = y + pyr[i].shape[0]
    for i in range(levels):
        rendered_per.append(np.pad(pyr[i], ( (0, x - pyr[i].shape[0]),(0, 0)), mode='constant'))
    im = np.concatenate(rendered_per, 1)
    return im


def display_pyramid(pyr, levels):
    """
    :param pyr: list of Gaussian or Laplacian pyramids of an image
    :param levels: the amount of levels of the pyramids to stack together horizontally
    :return: plots the pyramids
    """
    plt.figure()
    plt.imshow(render_pyramid(pyr, levels), cmap=plt.get_cmap('gray'))


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: grayscale image with values in [0,1]
    :param im2: grayscale image with values in [0,1]
    :param mask: boolian image, true means value from im1, false means value from im2
    :param max_levels: amount of pyramids to construct for each image
    :param filter_size_im: size of the Gaussian filter for the images blur
    :param filter_size_mask: size of the Gaussian filter for the mask blur
    :return: blended image of im1 and im2 , the blended area depends on the mask
    """
    l1, vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, _ = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    g_m, _ = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    l_out = []
    for i in range(len(l1)):
        l_out.append(np.multiply(l1[i], g_m[i]) + np.multiply(l2[i], (np.ones(g_m[i].shape) - g_m[i])))
    ret = laplacian_to_image(l_out, vec, np.ones(len(l_out), dtype=np.float64))
    return np.clip(ret, a_min=0, a_max=1)


def blending_example1():
    one = read_image(relpath('externals/first_im1.jpg'), 2)
    two = read_image(relpath('externals/first_im2.jpg'), 2)
    mask = read_image(relpath('externals/first_mask.jpg'), 1)
    out = np.ndarray(shape=one.shape)
    out[:, :, 0] = pyramid_blending(one[:, :, 0], two[:, :, 0], mask, 4, 3, 3)
    out[:, :, 1] = pyramid_blending(one[:, :, 1], two[:, :, 1], mask, 4, 3, 3)
    out[:, :, 2] = pyramid_blending(one[:, :, 2], two[:, :, 2], mask, 4, 3, 3)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(one, cmap=plt.get_cmap('gray'))
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(two, cmap=plt.get_cmap('gray'))
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(mask, cmap=plt.get_cmap('gray'))
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(out, cmap=plt.get_cmap('gray'))
    plt.show()
    return one, two, mask.astype(np.bool), out


def blending_example2():
    one = read_image(relpath('externals/second_im1.jpg'), 2)
    two = read_image(relpath('externals/second_im2.jpg'), 2)
    mask = read_image(relpath('externals/second_mask.jpg'), 1)
    out = np.ndarray(shape=one.shape)
    out[:, :, 0] = pyramid_blending(one[:, :, 0], two[:, :, 0], mask, 3, 5, 3)
    out[:, :, 1] = pyramid_blending(one[:, :, 1], two[:, :, 1], mask, 3, 5, 3)
    out[:, :, 2] = pyramid_blending(one[:, :, 2], two[:, :, 2], mask, 3, 5, 3)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(one, cmap=plt.get_cmap('gray'))
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(two, cmap=plt.get_cmap('gray'))
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(mask, cmap=plt.get_cmap('gray'))
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(out, cmap=plt.get_cmap('gray'))
    plt.show()
    return one, two, mask.astype(np.bool), out
