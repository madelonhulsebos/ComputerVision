import numpy as np

from skimage.transform import resize
from skimage import io
from sklearn.externals import joblib


# Compute integral image
def int_img(im_array, width, height, channels):

    int_image = np.zeros((width, height, channels))

    # Initialize the corner pixel value
    int_image[0, 0, 0] = im_array[0, 0, 0]
    int_image[0, 0, 1] = im_array[0, 0, 1]
    int_image[0, 0, 2] = im_array[0, 0, 2]
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            for c in range(channels):
                int_image[x, y, c] = int(im_array[x - 1, y - 1, c]) + int(im_array[x - 1, y - 1, c]) + int(
                                     im_array[x, y, c])

    return int_image


# Read image
img = io.imread(r'C:\Users\Madelon\Documents\Madelon\1. TU Delft\CS\1. MSc 1\Q4 Computer Vision\Project\ComputerVision\datasets\cars_markus\image_0001.jpg')

im_array = np.array(img)

# Dimensions of the image
width = im_array.shape[0]
height = im_array.shape[1]
channels = im_array.shape[2]

# Compute integral image
integral_image = int_img(im_array, width, height, channels)

# Load the trained classifier
bdt = joblib.load(r'C:\Users\Madelon\Documents\Madelon\1. TU Delft\CS\1. MSc 1'
                  r'\Q4 Computer Vision\Project\ComputerVision\License plate detection/classifiers/vj_bdt')

# Apply classifier

# Take [w:120, h:60] as dimensions of the sliding windows
# Resample image 2 times with same window
windowsize = [120, 60]

# Resize image to scan image on different scale
img_1 = resize(img, 0.8)
img_2 = resize(img_1, 0.8)






