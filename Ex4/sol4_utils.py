import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d


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


def create_kernel(kernel_size, dim):
    """
    Helper function that creates a 1D\2D Gaussian kernel
    :param kernel_size: len of the desired kernel
    :param dim: 1D or 2D kernel
    :return: 1D\2D Gaussian kernel
    """
    if kernel_size == 1:
        return np.ones(shape=(1, 1))
    base = np.asarray([1, 1])
    ker = base
    for i in range(2, kernel_size):
        ker = np.convolve(ker, base)
    ker = ker.reshape((kernel_size, 1))
    if dim == 2:
        ker = convolve2d(ker, ker.T)
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
    filter_vec = create_kernel(filter_size, 1)
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

def blur_spatial(im, kernel_size):
    """
    :param im: N * K image with values in [0,1] float64
    :param kernel_size: len of the row\col of the desired kernel
    :return: blurred image that was created via convolution between the image and a Gaussian kernel
    """
    ker = create_kernel(kernel_size, 2)
    return convolve2d(im, ker, mode='same', boundary='symm')