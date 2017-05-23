import numpy as np

from skimage import io
from os import walk

from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


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


# List all filenames of directory
dir_list = []
for (dirpath, dirnames, filenames) in walk('datasets/acme_licenses'):
    dir_list.extend(filenames)
    break

# Read all images from directory through the list of filenames
im_list = []
for file in dir_list :
    img = io.imread('datasets/acme_licenses/' + file)
    im_list.append(img)

# Show first image from the list
# io.imshow(im_list[0])

# Compute Haar-features
count = 0
numimages = 10
haar_feature1 = np.zeros((numimages, 3))
haar_feature2 = np.zeros((numimages, 3))
for i in im_list[0:numimages] :

    img = i
    im_array = np.array(img)

    width = img.shape[0]
    height = img.shape[1]
    channels = img.shape[2]

    # Compute integral image
    integral_image = int_img(im_array, width, height, channels)
    adj_height = height - 1
    adj_width = width - 1

    # Compute Haar-feature values for RGB color channels
    for c in range(channels) :
        lower_box = int(im_array[adj_width, adj_height, c]) - int(im_array[adj_width, int(height/2), c]) \
                    - int(im_array[0, adj_height, c]) + int(im_array[0, int(height/2), c])
        upper_box = int(im_array[adj_width, int(height/2), c])
        haar_feature1[count, c] = upper_box - lower_box
        haar_feature2[count, c] = lower_box - upper_box # This comes down to '- haar_feature1'

    count = count + 1


# Untrained model
# bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                         algorithm="SAMME",
#                         n_estimators=200)

# Train model on training data
# bdt.fit(X_train, y_train)

# Save model for further use
# joblib.dump(bdt, r'C:\Users\Madelon\Documents\Madelon\1. TU Delft\CS\1. MSc 1'
#                 r'\Q4 Computer Vision\Project\ComputerVision\License plate detection/classifiers/vj_bdt')