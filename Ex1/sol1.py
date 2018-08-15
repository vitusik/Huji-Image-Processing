import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray

# The transform matrix from RGB plane to YIQ plane as given in the TA session
transform_matrix = np.array([[.299, .587, .114], [.596, -.275, -.321], [.212, -.523, .311]])


inverse_transform_matrix = np.linalg.inv(transform_matrix)


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


def imdisplay(filename, representation):
    """
    displays the input
    :param filename: path to the image we want to display
    :param representation: 1 means the image needs to be transformed to grayscale, 0 leaves the image as is
    """
    plt.figure()
    im = read_image(filename, representation)
    im = np.clip(im, 0, 1)
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.show()


def rgb2yiq(imRGB):
    """
    :param imRGB: numpy array with values between 0 and 1 that represents the image in RGB plane
    :return: numpy array that represents the image in YIQ plane
    """
    return np.dot(imRGB, transform_matrix.transpose())


def yiq2rgb(imYIQ):
    """
    :param imYIQ: numpy array with values between 0 and 1 that represents the image in YIQ plane
    :return: numpy array that represents the image in RGB plane
    """
    return np.dot(imYIQ, inverse_transform_matrix.transpose())


def histogram_equalize(im_orig):
    """
    function that equalizes the histogram of the given image
    :param im_orig: image whose histogram we want to equalize,
                    which is numpy array with values between 0 and 1 that represents the image in RGB\grayscale plane
    :return: list[im_eq, hist_orig, hist_eq]
            im_eq: is the equalized image, numpy array with values of 0-1
            hist_orig: is the original histogram
            hist_eq: is the histogram of the equalized image
    """
    rgb = False
    # grayscale images have only 2 dimensions
    if len(im_orig.shape) == 3:
        im_orig_yiq = rgb2yiq(im_orig)
        # working only on the Y channel of the image
        work_im = im_orig_yiq[:, :, 0]
        rgb = True
    else:
        work_im = im_orig
    hist_orig, bins = np.histogram(work_im.flatten(), bins=256)
    # cumulative histogram of the image
    cum_hist = hist_orig.cumsum()
    # lut as seen in class
    lut = np.round(255 * (cum_hist - cum_hist.min())/(cum_hist[-1] - cum_hist.min()))
    # remapping the pixels intensities
    eq_img = np.interp(work_im, bins[0:-1], lut)
    eq_img.reshape(work_im.shape)
    eq_img /= 255
    if rgb:
        # change the Y channel in the original image
        im_orig_yiq[:, :, 0] = eq_img[:, :]
        im_eq = yiq2rgb(im_orig_yiq)
        im_eq = np.clip(im_eq,0,1)
    else:
        im_eq = eq_img
    hist_eq, bins2 = np.histogram(im_eq.flatten(), bins=256)
    return [im_eq, hist_orig, hist_eq]


def quantize(img_orig, n_quant, n_iter):
    """
    function that quantize an image
    :param img_orig: image which we want to quantize,
                    which is numpy array with values between 0 and 1 that represents the image in RGB\grayscale plane
    :param n_quant: amount of different colors we want in the output
    :param n_iter: amount of iterations the process should run
    :return: list[im_quant, error]
                im_quant: the quantized image, which is a numpy array that represents the image in RGB\grayscale plane
                error: array in the length of at most n_iter, which holds the mistake in each iteration
    """
    rgb = False
    if len(img_orig.shape) == 3:
        im_orig_yiq = rgb2yiq(img_orig)
        work_im = im_orig_yiq[:, :, 0]
        rgb = True
    else:
        work_im = img_orig

    z = np.ndarray(shape=n_quant+1,dtype=int)
    q = np.ndarray(shape=n_quant, dtype=int)
    error = []
    init_err = 0
    hist, bins = np.histogram(work_im, 256)
    hist2 = hist * np.linspace(0,255,256)
    cum = hist.cumsum()
    # ppb - pixels per bin
    ppb = round(cum[-1] / n_quant)
    z[0] = 0
    z[n_quant] = 255
    for i in range(1, n_quant):
        # for each quant go over the cumulative histogram and look for the last index that holds the condition
        res = np.array(np.where(cum <= ppb * i))
        z[i] = (res[0])[-1]
    for i in range(n_quant):
        # according to what we've seen in class
        q[i] = (np.cumsum(hist2[z[i]:z[i+1]])[-1]) / (np.cumsum(hist[z[i]:z[i+1]])[-1])
    for i in range(n_quant):
        # according to what we've seen in class
        init_err += np.cumsum(pow((q[i] * hist - hist2)[z[i]:z[i+1]],2))[-1]
    cur_err = init_err

    for iter in range(n_iter):
        new_err = 0
        new_z = np.ndarray(shape=n_quant + 1, dtype=int)
        new_z[0] = 0
        new_z[n_quant] = 255
        new_q = np.ndarray(shape=n_quant, dtype=int)
        for i in range(1,n_quant):
            new_z[i] = (q[i-1] + q[i]) / 2
        for i in range(n_quant):
            new_q[i] = (np.cumsum(hist2[new_z[i]:new_z[i + 1]])[-1]) / (np.cumsum(hist[new_z[i]:new_z[i + 1]])[-1])
        for i in range(n_quant):
            new_err += np.cumsum(pow((new_q[i] * hist - hist2)[new_z[i]:new_z[i + 1]], 2))[-1]
        if cur_err > new_err:
            error.append(new_err)
            z = new_z
            q = new_q
            cur_err = new_err
        else:
            error.append(cur_err)
            # the mistake stayed the same, meaning the function has converged
            break
    lut = np.zeros(256)
    for i in range(n_quant):
        lut[z[i]:z[i + 1]] = q[i]
    quant_im = np.interp(work_im.flatten(), np.linspace(0, 1, 256), lut)
    quant_im = quant_im.reshape(work_im.shape)
    quant_im /= 255
    if rgb:
        im_orig_yiq[:, :, 0] = quant_im[:, :]
        im_quant = yiq2rgb(im_orig_yiq)
    else:
        im_quant = quant_im
    return [im_quant, error]


