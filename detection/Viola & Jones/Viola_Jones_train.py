import matplotlib.pyplot as plt
import numpy as np
import os

from os import walk
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#Suppress tensorflow compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compute feature vector corresponding to the Haar-like feature focusing on the license plate characters
def midchar_features(int_im, h, w) :
    # Adjust height and width of image to Python origin (0,0)
    w = w - 1
    h = h - 1
    frame_width = 20

    featurevalues = np.zeros((int(np.ceil(w/frame_width))))
    count1 = 1

    # First feature frame
    featurevalues[0] = int(int_im[int(2*np.ceil(h/3)), frame_width]) - int(int_im[int(np.floor(h/3)), frame_width]) \
                         - int(int_im[int(h), frame_width])

    # Takes 6 as stepsize (framewidth of feature)
    for x in range(frame_width, w - frame_width, frame_width) :
        cornerx = x # Save the remaining number of pixels to be framed in a feature
        black_box1 = int(int_im[int(np.floor(h/3)), x+frame_width]) -  int(int_im[int(np.floor(h/3)), x-1]) # upper black box
        white_box = int(int_im[int(2*np.ceil(h/3)), x+frame_width]) - int(int_im[int(2*np.ceil(h/3)), x-1]) \
                    -  int(int_im[int(np.floor(h/3)), x+frame_width]) + int(int_im[int(np.floor(h/3)), x-1])
        black_box2 = int(int_im[int(h), x+frame_width]) - int(int_im[int(h), x-1]) - int(int_im[int(2*np.ceil(h/3)), x+frame_width]) \
                     + int(int_im[int(2*np.ceil(h/3)), x-1]) # lower black box

        featurevalues[count1] = white_box - black_box1 - black_box2
        count1 = count1 + 1

    # Last feature frame
    featurevalues[count1] = int(int_im[int(h), cornerx]) + int(int_im[ int(np.ceil(h/3)), cornerx - 1]) \
                              + int(int_im[int(2*np.ceil(h/3)), w]) - int(int_im[int(h), cornerx - 1])

    return featurevalues


# Compute featurevector corresponding to the Haar-like feature focusing on a 'small' centered horizontal rectangle
# aimed at capturing the entire center range of the license plate
def midplate_feature(int_im, h, w) :
    # Adjust height and width of image to Python origin (0,0)
    h = h - 1
    w = w - 1

    white_box = int(int_im[int(2*np.ceil(h/3)), w]) - int(int_im[int(np.floor(h/3)), w])
    black_box1 = int(int_im[int(np.floor(h/3)), w])
    black_box2 = int(int_im[h, w]) - int(int_im[int(2*np.ceil(h/3)), w]) - int(int_im[int(np.floor(h/3)), w])
    featurevalue = white_box - black_box1 - black_box2

    return featurevalue


### Training data
positive_images = []
for (__, __, filename) in walk('datasets/acme_licenses/centered_opaque_rotated_background') :
    positive_images.extend(filename)
    break

negative_images = []
for (__, __, filename) in walk('datasets/negative_instances'):
    negative_images.extend(filename)
    break

feature_height = 80
feature_width = 160
frame_width = 20
num_features = int(np.ceil((feature_width / frame_width)) + 1)

trainset = np.empty((0,num_features))
integral_image = np.zeros((feature_height, feature_width))
for filename in positive_images :
    img = rgb2gray(io.imread('datasets/acme_licenses/centered_opaque_rotated_background/' + filename))
    img = np.array(resize(img, (feature_height, feature_width)))
    np.cumsum(img, axis=1, out=integral_image)
    feature_vector = np.hstack([midchar_features(integral_image, feature_height, feature_width), midplate_feature(integral_image, feature_height, feature_width)])
    trainset = np.vstack((trainset, feature_vector))

for filename in negative_images :
    img = rgb2gray(io.imread('datasets/negative_instances/' + filename))
    img = np.array(resize(img, (feature_height, feature_width)))
    np.cumsum(img, axis=1, out=integral_image)
    feature_vector = np.hstack([midchar_features(integral_image, feature_height, feature_width), midplate_feature(integral_image, feature_height, feature_width)])
    trainset = np.vstack((trainset, feature_vector))

trainlabels = np.concatenate((np.ones(len(positive_images)), np.zeros(len(negative_images))), axis=0)

# Untrained model
# Todo: check for optimal parameters AdaBoost
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

# Potentially apply PCA on data
pca = PCA(n_components = 0.90) # Or 0.90/0.85 ...
trainset_pca = pca.fit_transform(trainset)

# Train AdaBoost model
bdt_nopca = bdt.fit(trainset, trainlabels)
bdt_pca = bdt.fit(trainset_pca, trainlabels)

# Save trained model for future use
joblib.dump(bdt_nopca, r'detection/classifiers/vj_bdt_nopca')
joblib.dump(bdt_pca, r'detection/classifiers/vj_bdt_pca')


# Test trained model on contextual test images
test_images = []
for (__, __, filename) in walk('datasets/cars_markus'):
    test_images.extend(filename)
    break

for filename in test_images[2:3]:

    integral_image = np.zeros((feature_height, feature_width))
    img = np.array(rgb2gray(io.imread('datasets/cars_markus/' + filename)))
    img_copy = io.imread('datasets/cars_markus/' + filename)

    # Generate windows
    window_border_color = [0]
    size = (feature_height, feature_width)
    x_steps = 10
    y_steps = 10

    for i in range(y_steps):
        y = int((img.shape[0] - size[0]) * i / y_steps)
        for j in range(x_steps):
            x = int((img.shape[1] - size[1]) * j / x_steps)
            segment = img[y:y + size[0], x:x + size[1]]
            np.cumsum(segment, axis=1, out=integral_image)
            feature_vector = np.hstack([midchar_features(integral_image, feature_height, feature_width),
                                        midplate_feature(integral_image, feature_height, feature_width)])
            test_sample = pca.transform(np.array(feature_vector)).reshape(1,-1)

            if bdt_pca.predict(test_sample) == 1.0:
                tmp = np.copy(img_copy)
                tmp[y:y+size[0], x] = window_border_color
                tmp[y, x:x+size[1]] = window_border_color
                tmp[y:y+size[0], x+size[1]] = window_border_color
                tmp[y+size[0], x:x+size[1]] = window_border_color

                plt.figure()
                plt.imshow(tmp, cmap = plt.get_cmap('gray'))
                plt.show()