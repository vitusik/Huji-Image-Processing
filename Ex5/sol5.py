from sol5_utils import *
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy.ndimage import map_coordinates, convolve
from keras.layers import Input, Activation, Convolution2D, merge
from keras.models import Model
from keras.optimizers import Adam
# our image cache
im_dic = {}


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


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    This function returns a generator object which outputs a tuple of (source_batch,target_batch)
    :param filenames: a list of filenames of clean images
    :param batch_size: the size of the batch of images for each iteration of SGD
    :param corruption_func: a function which receives a numpy array of an image, and returns randomly generated
                            corrupted image
    :param crop_size: tuple(height, width) specifying the crop size of the extracted patches
    :return: a python generator object which outputs a tuple of (source_batch,target_batch)
    """
    while True:
        source = np.ndarray(shape=(batch_size, 1, crop_size[0], crop_size[1]))
        target = np.ndarray(shape=(batch_size, 1, crop_size[0], crop_size[1]))
        for i in range(batch_size):
            im_index = np.random.choice(len(filenames))
            if filenames[im_index] not in im_dic:
                im_dic[filenames[im_index]] = read_image(filenames[im_index], 1)
            cur_im = im_dic[filenames[im_index]]
            corrupt_im = corruption_func(cur_im)
            # insuring that the window will not get out of the images bounds
            rnd_h = np.random.choice(cur_im.shape[0] - crop_size[0])
            rnd_w = np.random.choice(cur_im.shape[1] - crop_size[1])
            h = np.arange(rnd_h, rnd_h + crop_size[0])
            w = np.arange(rnd_w, rnd_w + crop_size[1])
            clean_patch = map_coordinates(input=cur_im, coordinates=np.meshgrid(h, w), order=1, prefilter=False)
            clean_patch = clean_patch.T
            clean_patch = clean_patch - 0.5
            corrupt_patch = map_coordinates(input=corrupt_im, coordinates=np.meshgrid(h, w), order=1, prefilter=False)
            corrupt_patch = corrupt_patch.T
            corrupt_patch = corrupt_patch - 0.5
            source[i] = corrupt_patch
            target[i] = clean_patch
        yield (source, target)


def resblock(input_tensor, num_channels):
    """
    Takes the input_tensor through a residual block structure and outputs the result
    :param input_tensor: input tensor for residual block
    :param num_channels: num of channels for each convolution layer
    :return: output tensor
    """
    intermediate_tensor = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    intermediate_tensor = Activation('relu')(intermediate_tensor)
    intermediate_tensor = Convolution2D(num_channels, 3, 3, border_mode='same')(intermediate_tensor)
    output_tensor = merge([input_tensor, intermediate_tensor], mode='sum')
    return output_tensor


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Creates a CNN in Resnet architecture
    :param height: height of the convolution filter
    :param width: width of the convolution filter
    :param num_channels: amount of filters to produce in each convolution layer
    :param num_res_blocks: amount of residual blocks in the neural net architecture
    :return: a Model object which represents our CNN
    """
    input_tensor = Input(shape=(1, height, width))
    intermediate_tensor = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    intermediate_tensor = Activation('relu')(intermediate_tensor)
    input_after_first_relu = intermediate_tensor
    for i in range(num_res_blocks):
        intermediate_tensor = resblock(intermediate_tensor, num_channels)
    intermediate_tensor = merge([input_after_first_relu, intermediate_tensor], mode='sum')
    output_tensor = Convolution2D(1, 3, 3, border_mode='same')(intermediate_tensor)
    model = Model(input=input_tensor, output=output_tensor)
    return model


def train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples):
    """
    Trains our CNN via the fit_generator method from the keras module
    :param model: general model of a NN model for image restoration
    :param images: list of path to images
    :param corruption_func: a function which receives a numpy array of an image, and returns randomly generated
                            corrupted image
    :param batch_size: the size of the batch of images for each iteration of SGD
    :param samples_per_epoch: amount of samples in each epoch
    :param num_epochs: number of epochs the training will run for
    :param num_valid_smaples: number of samples to test on after each epoch
    """
    model_shape = model.input_shape
    crop_tuple = (model_shape[2], model_shape[3])
    train_len = int(len(images) * 0.8)
    train_images = images[0:train_len]
    valid_images = images[train_len:-1]
    train_gen = load_dataset(train_images, batch_size, corruption_func, crop_tuple)
    valid_gen = load_dataset(valid_images, num_valid_samples, corruption_func, crop_tuple)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(generator=train_gen, samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=valid_gen, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    Restores a corrupted image via a trained CNN
    :param corrupted_image: the corrupted image we wish to restore
    :param base_model: the CNN which will restore the image
    :return: an uncorrupted image
    """
    h, w = corrupted_image.shape
    i = Input(shape=(1, h, w))
    o = base_model(i)
    cur_model = Model(input=i, output=o)
    x = corrupted_image - 0.5
    x = np.reshape(x, newshape=(1, x.shape[0], x.shape[1]))
    y = cur_model.predict(x[np.newaxis])[0]
    y = y + 0.5
    y = np.clip(y, a_min=0, a_max=1).astype(np.float64)
    return y.reshape(y.shape[1], y.shape[2])


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Adds zero mean with unknown standard deviation noise to an input image
    :param image: the image we wish to corrupt
    :param min_sigma: the lower bound of the standard deviation of the noise
    :param max_sigma: the upper bound of the standard deviation of the noise
    :return: corrupted image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, size=(image.shape[0], image.shape[1]))
    corrupted = noise + image
    corrupted = corrupted * 255
    corrupted = np.around(corrupted)
    corrupted = corrupted / 255
    return np.clip(corrupted, a_min=0, a_max=1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    Trains a CNN with Resnet architecture to denoise images
    :param num_res_blocks: amount of resblocks in the architecture
    :param quick_mode: mode for testing
    :return: trained CNN
    """
    denoise_model = build_nn_model(24, 24, 48, num_res_blocks)
    if quick_mode:
        train_model(denoise_model, images_for_denoising(), lambda image: add_gaussian_noise(image, 0, 0.2), 10, 30,
                    2, 30)
    else:
        train_model(denoise_model, images_for_denoising(), lambda image: add_gaussian_noise(image, 0, 0.2), 100, 10000,
                    5, 1000)
    return denoise_model


def add_motion_blur(image, kernel_size, angle):
    """
    Adds a motion blur to an image
    :param image: the image we wish to blur
    :param kernel_size: size of the blur kernel
    :param angle: the angle in [0,pi) of the blur
    :return: blurred image of the input
    """
    blur = motion_blur_kernel(kernel_size, angle)
    corrupted_image = convolve(image, blur, mode='nearest')
    corrupted_image = corrupted_image * 255
    corrupted_image = np.around(corrupted_image)
    corrupted_image = corrupted_image / 255
    return np.clip(corrupted_image, a_min=0, a_max=1)


def random_motion_blur(image, list_of_kernel_size):
    """
    Adds a random generated motion blur to an image
    :param image: the image we wish to blur
    :param list_of_kernel_size: list of possible kernels
    :return: blurred image of the input
    """
    kernel_index = int(np.random.uniform(0, len(list_of_kernel_size) - 1))
    angle = np.random.uniform(0, np.pi)
    return add_motion_blur(image, list_of_kernel_size[kernel_index], angle)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    Trains a CNN with Resnet architecture to deblur images
    :param num_res_blocks: amount of resblocks in the architecture
    :param quick_mode: mode for testing
    :return: trained CNN
    """
    deblur_model = build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        train_model(deblur_model, images_for_denoising(), lambda image: random_motion_blur(image, [7]), 10, 30,
                    2, 30)
    else:
        train_model(deblur_model, images_for_deblurring(), lambda image: random_motion_blur(image, [7]), 100, 10000,
                    10, 1000)
    return deblur_model



