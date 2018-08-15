import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy.signal import convolve2d

import matplotlib.pyplot as plt

x_con = np.asarray([1, 0, -1]).reshape(1, 3)
y_con = x_con.T


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


def DFT(signal):
    """
    :param signal: N * K matrix with values in [0,1] float64
    :return: the Fourier transform of the columns
    """
    size = signal.shape[0]
    n = np.arange(size)
    k = n.reshape((signal.shape[0], 1))
    four_mat = np.exp(-2j * np.pi * k * n / size)
    return np.dot(four_mat, signal)


def IDFT(fourier_siganl):
    """
    :param fourier_siganl: N * K matrix of values in Fourier space
    :return: the inverse Fourier transform of the columns
    """
    size = fourier_siganl.shape[0]
    n = np.arange(size)
    k = n.reshape((fourier_siganl.shape[0], 1))
    inverse_four_mat = np.exp(2j * np.pi * k * n / size)
    inverse_four_mat /= size
    sig = np.dot(inverse_four_mat, fourier_siganl)
    return sig


def DFT2(image):
    """
    :param image:  N * K image with values in [0,1] float64
    :return: the Fourier transform of the iamge
    """
    fourier = np.zeros(shape=image.shape, dtype=np.complex128)
    fourier = DFT(image)
    fourier = DFT(fourier.T)
    return fourier.T


def IDFT2(fourier_image):
    """
    :param fourier_image: N * K image of values in Fourier space
    :return: the inverse Fourier transform of the image
    """
    im = np.zeros(shape=fourier_image.shape, dtype=np.complex128)
    im = IDFT(fourier_image)
    im = IDFT(im.T)
    return im.T


def conv_der(image):
    """
    Calculating the derivative via convolution
    :param image: N * K image with values in [0,1] of float64
    :return: N * K matrix which holds the derivative values of the original image
    """
    der_x = convolve2d(image, x_con, mode='same')
    der_y = convolve2d(image, y_con, mode='same')
    mag = np.sqrt(der_x**2 + der_y**2)
    return mag


def fourier_der(im):
    """
    Calculating the derivative via Fourier transform
    :param image: N * K image with values in [0,1] of float64
    :return: N * K matrix which holds the derivative values of the original image
    """
    four_im = DFT2(im)
    x = np.asarray(np.arange(im.shape[0]))
    x = np.fft.fftshift(x)
    y = np.asarray(np.arange(im.shape[1]))
    y = np.fft.fftshift(y)
    four_im_x = four_im.T * x
    four_im_x = four_im_x.T
    four_im_y = four_im * y
    der_x = IDFT2(four_im_x)
    der_x *= (2j * np.pi/(im.shape[0]))
    der_y = IDFT2(four_im_y)
    der_y *= (2j * np.pi/(im.shape[1]))
    mag = np.real(np.sqrt(der_x ** 2 + der_y ** 2))
    mag = mag.astype(np.float64)
    return mag


def create_kernel(kernel_size):
    """
    Helper function that creates a 2D Gaussian kernel
    :param kernel_size: len of the row\col of the desired kernel
    :return: Gaussian kernel with shape of (kernel_size, kernel_size)
    """
    if kernel_size == 1:
        return np.ones(shape=(1, 1))
    base = np.asarray([1, 1])
    ker = base
    for i in range(2, kernel_size):
        ker = np.convolve(ker, base)
    ker = ker.reshape((1, ker.shape[0]))
    ker = convolve2d(ker, ker.T)
    return ker / np.cumsum(ker)[-1]


def blur_spatial(im, kernel_size):
    """
    :param im: N * K image with values in [0,1] float64
    :param kernel_size: len of the row\col of the desired kernel
    :return: blurred image that was created via convolution between the image and a Gaussian kernel
    """
    ker = create_kernel(kernel_size)
    return convolve2d(im, ker, mode='same', boundary='fill')


def pad_ker(ker, shape):
    """
    Helper function that pads a given ker with 0 around it to strech it to the given shape
    :param ker: 2D Gaussian kernel
    :param shape: the wanted dimensions of the kernel
    :return: matrix in the shape of the input with the input kernel at the middle of it
    """
    x = (int)((shape[0] / 2) - (ker.shape[0] / 2))
    y = (int)((shape[1] / 2) - (ker.shape[1] / 2))
    pad = ((x, shape[0] - x - ker.shape[0]), (y, shape[1] - y - ker.shape[1]))
    return np.pad(ker, pad, mode='constant', constant_values=0)


def blur_fourier(im, kernel_size):
    """
    :param im: N * K image with values in [0,1] float64
    :param kernel_size: len of the row\col of the desired kernel
    :return: blurred image that was created via a multiplication between the image in Fourier space and the kernel in
             Fourier space
    """
    ker = create_kernel(kernel_size)
    ker = pad_ker(ker, im.shape)
    ker = np.fft.fftshift(ker)
    ker = DFT2(ker)
    fourier = DFT2(im)
    fourier = np.multiply(fourier, ker)
    ret_im = IDFT2(fourier)
    ret_im = np.real(ret_im)
    ret_im = ret_im.astype(np.float64)
    return ret_im



