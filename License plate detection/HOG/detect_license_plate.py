from os import walk
import numpy as np
import pandas as pd
import scipy
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def extract_hog(img):
    return hog(rgb2gray(resize(img,
                               (40, 80),
                               mode='constant')),
               orientations=16,
               pixels_per_cell=(8, 8),
               cells_per_block=(4, 4),
               block_norm='L2')

positive_dir = '../../datasets/acme_licenses'
negative_dir = '../../datasets/negative_instances'
testing_dir = '../../datasets/cars_markus'
matches_dir = 'matches'

_, _, positive_images = next(walk(positive_dir))
_, _, negative_images = next(walk(negative_dir))

X = np.concatenate((
    np.array([extract_hog(imread('%s/%s' % (positive_dir, filename))) for filename in positive_images]),
    np.array([extract_hog(imread('%s/%s' % (negative_dir, filename))) for filename in negative_images])
))

Y = np.concatenate((np.ones(len(positive_images)), np.zeros(len(negative_images))))

# Apply PCA
pca = PCA(n_components=.99)
X = pca.fit_transform(X)

print('Number of features: %d' % X.shape[1])

classifier = SVC()

# print "Initializing cross-validation"
# kf = KFold(n_splits=10, shuffle=True)
# for train_index, test_index in kf.split(X):
#     classifier.fit(X[train_index], Y[train_index])
#     print 'Accuracy for current iteration: %0.2f' % (classifier.score(X[test_index], Y[test_index]))

classifier.fit(X, Y)

for (_, _, images) in walk(testing_dir):
    break

for filename in images:
    print("Analyzing %s" % filename)

    img = imread('%s/%s' % (testing_dir, filename))

    # Generate windows
    window_border_color = [255, 0, 0]
    size = (100, 200)
    stepSize = 16

    # TopLeft, TopRight, BottomLeft, BottomRight
    cum_matches = [0, 0]
    num_matches = 0

    for i in range(int((img.shape[0] - size[0]) / stepSize)):
        y = i * stepSize
        for j in range(int((img.shape[1] - size[1]) / stepSize)):
            x = j * stepSize

            segment = img[y:y+size[0], x:x+size[1], :]
            data = pca.transform(np.array([extract_hog(segment)]))

            if classifier.predict(data) == 1.0:
                tmp = np.copy(img)
                tmp[y:y+size[0], x, :] = window_border_color
                tmp[y:y+size[0], x+size[1], :] = window_border_color
                tmp[y, x:x+size[1], :] = window_border_color
                tmp[y+size[0], x:x+size[1], :] = window_border_color

                # plt.imshow(tmp)
                # plt.show()

                cum_matches[0] += y
                cum_matches[1] += x

                num_matches += 1

    avg_match = None if num_matches == 0 else (cum_matches[0] / num_matches,
                                               cum_matches[1] / num_matches)
    if avg_match is not None:
        y = int(avg_match[0])
        x = int(avg_match[1])

        segment = img[y:y+size[0], x:x+size[1], :]

        scipy.misc.imsave('%s/%s' % (matches_dir, filename), segment.astype(np.uint8))
