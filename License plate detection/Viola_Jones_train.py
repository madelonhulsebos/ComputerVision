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
def midchar_features(int_im, w, h) :
    # Adjust height and width of image to Python origin (0,0)
    w = w - 1
    h = h - 1

    featurevalues = np.zeros((int(np.ceil(w/4))))
    count1 = 1

    # First feature frame
    featurevalues[0] = int(int_im[6,int(2*np.ceil(h/3))]) - int(int_im[6, int(np.floor(h/3))]) \
                         - int(int_im[6, int(h)])

    # Takes 6 as stepsize (framewidth of feature)
    for x in range(6, w - 6, 6) :
        cornerx = x # Save the remaining number of pixels to be framed in a feature
        black_box1 = int(int_im[x+6, int(np.floor(h/3))]) -  int(int_im[x-1, int(np.floor(h/3))]) # upper black box
        white_box = int(int_im[x+6, int(2*np.ceil(h/3))]) - int(int_im[x-1, int(2*np.ceil(h/3))]) \
                    -  int(int_im[x+6, int(np.floor(h/3))]) + int(int_im[x-1, int(np.floor(h/3))])
        black_box2 = int(int_im[x+6, int(h)]) - int(int_im[x-1, int(h)]) - int(int_im[x+6, int(2*np.ceil(h/3))]) \
                     + int(int_im[x-1, int(2*np.ceil(h/3))]) # lower black box

        featurevalues[count1] = white_box - black_box1 - black_box2
        count1 = count1 + 1

    # Last feature frame
    featurevalues[count1] = int(int_im[cornerx, int(h)]) + int(int_im[cornerx - 1, int(np.ceil(h/3))]) \
                              + int(int_im[w, int(2*np.ceil(h/3))]) - int(int_im[cornerx - 1, int(h)])

    return featurevalues


# Compute featurevector corresponding to the Haar-like feature focusing on a 'small' centered horizontal rectangle
# aimed at capturing the entire center range of the license plate
def midplate_feature(int_im, w, h) :
    # Adjust height and width of image to Python origin (0,0)
    w = w - 1
    h = h - 1

    white_box = int(int_im[w, int(2*np.ceil(h/3))]) - int(int_im[w, int(np.floor(h/3))])
    black_box1 = int(int_im[w, int(np.floor(h/3))])
    black_box2 = int(int_im[w, h]) - int(int_im[w, int(2*np.ceil(h/3))]) - int(int_im[w, int(np.floor(h/3))])
    featurevalue = white_box - black_box1 - black_box2

    return featurevalue


### Training data
positive_images = []
for (__, __, filename) in walk('datasets/acme_licenses/centered_opaque_background') :
    positive_images.extend(filename)
    break

negative_images = []
for (__, __, filename) in walk('datasets/negative_instances'):
    negative_images.extend(filename)
    break

trainset = np.empty((0,51))
integral_image = np.zeros((200, 100))
for filename in positive_images :
    img = rgb2gray(io.imread('datasets/acme_licenses/centered_opaque_background/' + filename))
    img = np.array(resize(img, (200, 100)))
    np.cumsum(img, axis=1, out=integral_image)
    feature_vector = np.hstack([midchar_features(integral_image, 200, 100), midplate_feature(integral_image, 200, 100)])
    trainset = np.vstack((trainset, feature_vector))

for filename in negative_images :
    img = rgb2gray(io.imread('datasets/negative_instances/' + filename))
    img = np.array(resize(img, (200, 100)))
    np.cumsum(img, axis=1, out=integral_image)
    feature_vector = np.hstack([midchar_features(integral_image, 200, 100), midplate_feature(integral_image, 200, 100)])
    trainset = np.vstack((trainset, feature_vector))

trainlabels = np.concatenate((np.ones(len(positive_images)), np.zeros(len(negative_images))), axis=0)

# Untrained model
# Todo: check for optimal parameters AdaBoost
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

# Potentially apply PCA on data
# Todo: check for improvements by applying PCA to data
pca = PCA(n_components = 0.95) # Or 0.90/0.85 ...
#trainset = pca.fit_transform(trainset)

# Train AdaBoost model
bdt.fit(trainset, trainlabels)

# Save trained model for future use
#joblib.dump(bdt, r'License plate detection/classifiers/vj_bdt')


# Test trained model on contextual test images
test_images = []
for (__, __, filename) in walk('datasets/cars_markus'):
    test_images.extend(filename)
    break

for filename in test_images[0:3]:

    integral_image = np.zeros((200, 100))
    img = np.array(rgb2gray(io.imread('datasets/cars_markus/' + filename)))
    img_copy = io.imread('datasets/cars_markus/' + filename)

    # Generate windows
    window_border_color = [0]
    size = (200,100)
    x_steps = 10
    y_steps = 10

    for i in range(x_steps):
        x = int((img.shape[0] - size[0]) * i / x_steps)
        for j in range(y_steps):
            y = int((img.shape[1] - size[1]) * j / y_steps)
            segment = img[x:x + size[0], y:y + size[1]]
            np.cumsum(segment, axis=1, out=integral_image)
            feature_vector = np.hstack([midchar_features(integral_image, 200, 100),
                                        midplate_feature(integral_image, 200, 100)]).reshape(1, -1)

            if bdt.predict(feature_vector) == 1.0:
                tmp = np.copy(img_copy)
                tmp[x:x+size[0], y] = window_border_color
                tmp[x, y:y+size[1]] = window_border_color
                tmp[x:x+size[0], y+size[1]] = window_border_color
                tmp[x+size[0], y:y+size[1]] = window_border_color

                plt.figure()
                plt.imshow(tmp, cmap = plt.get_cmap('gray'))
                plt.show()