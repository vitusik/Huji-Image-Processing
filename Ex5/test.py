import matplotlib.pyplot as plt
#from sol5 import *
import numpy as np

'''
x = [1, 2, 3, 4, 5]

y = [0.0149, 0.011, 0.0097, 0.0083, 0.008]

plt.figure()
plt.plot(x,y)
plt.savefig('depth_plot_deblur.png')
'''
x = [1, 2, 3, 4, 5]
y = [0.0022, 0.0020, 0.0019, 0.0019, 0.0019]

plt.figure()
plt.plot(x,y)
plt.savefig('depth_plot_denoise.png')
'''


list_ima = images_for_denoising()

for i in range(1,6):
    test_model = learn_denoising_model(i, False)
    im = read_image(list_ima[0], 1)
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.show()

    im = add_gaussian_noise(im, 0, 0.2)
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.show()

    im = restore_image(im, test_model)
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.show()
'''



'''
list_ima = images_for_deblurring()

for i in range(1, 6):
    test_model = learn_deblurring_model(i, False)

    im = read_image(list_ima[0], 1)
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.show()

    im = random_motion_blur(im, [7])
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.show()

    im = restore_image(im, test_model)
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.show()

'''
