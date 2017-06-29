import numpy as np

from os import walk
from sklearn.externals import joblib
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC


# Extract HOG features from images
def extract_hog(img):
    return hog(rgb2gray(resize(img,
                               (40, 80),
                               mode='constant')),
               orientations=16,
               pixels_per_cell=(8, 8),
               cells_per_block=(4, 4),
               block_norm='L2')

# Extract images with and without license plate
positive_dir = '../../datasets/acme_licenses'
negative_dir = '../../datasets/negative_instances'


_, _, positive_images = next(walk(positive_dir))
_, _, negative_images = next(walk(negative_dir))

# Construct train dataset and labels
X = np.concatenate((
    np.array([extract_hog(imread('%s/%s' % (positive_dir, filename))) for filename in positive_images]),
    np.array([extract_hog(imread('%s/%s' % (negative_dir, filename))) for filename in negative_images])
))
Y = np.concatenate((np.ones(len(positive_images)), np.zeros(len(negative_images))))

# Train and fit classifier
classifier = SVC(C=10**-2, kernel='linear')
classifier.fit(X, Y)

# Save trained models
clsf_filename = 'detection_classifier_linear'
joblib.dump(classifier, clsf_filename)
