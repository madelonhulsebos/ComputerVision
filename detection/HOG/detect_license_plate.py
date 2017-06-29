import numpy as np

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.externals import joblib


def extract_hog(img):
    resized_image = resize(img, (40, 80), mode='constant')
    gray_image = rgb2gray(resized_image)

    return hog(gray_image,
               orientations=16,
               pixels_per_cell=(8, 8),
               cells_per_block=(4, 4),
               block_norm='L2')


def detect_plate(image):
    base_dir = 'detection/HOG/'

    classifier = joblib.load(base_dir + 'detection_classifier_linear')

    # Generate windows
    window_border_color = [255, 0, 0]
    size = (100, 200)
    stepSize = 16

    cum_matches = [0, 0]
    num_matches = 0

    # Use classifier to evaluate whether a window could represent a license plate
    for i in range(int((image.shape[0] - size[0]) / stepSize)):
        y = i * stepSize
        for j in range(int((image.shape[1] - size[1]) / stepSize)):
            x = j * stepSize

            # Apply Principal Component Analysis to window frame
            segment = image[y:y+size[0], x:x+size[1], :]
            data = np.array([extract_hog(segment)])

            if classifier.predict(data) == 1.0:
                tmp = np.copy(image)
                tmp[y:y+size[0], x, :] = window_border_color
                tmp[y:y+size[0], x+size[1], :] = window_border_color
                tmp[y, x:x+size[1], :] = window_border_color
                tmp[y+size[0], x:x+size[1], :] = window_border_color

                cum_matches[0] += y
                cum_matches[1] += x

                num_matches += 1

    # Average positive windows to come to a final decision of a positive frame
    avg_match = None if num_matches == 0 else (cum_matches[0] / num_matches,
                                               cum_matches[1] / num_matches)

    if avg_match is not None:
        y = int(avg_match[0])
        x = int(avg_match[1])

        segment = image[y:y+size[0], x:x+size[1], :]
    else:
        segment = image

    return segment
