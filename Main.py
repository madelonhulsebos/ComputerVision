from os import walk
from skimage.io import imread, imsave

from detection.HOG.detect_license_plate import detect_plate
from segmentation.Segmentation import segment_license_plate, segment_characters
from recognition.cnn.test import classify_characters

import matplotlib.pyplot as plt
import re
import sys

testing_dir = 'datasets/cars_markus'

for (_, _, images) in walk(testing_dir):
    break

label_file = open("labels.txt", "r")
labels = label_file.read().split("\n")

# result_file = open("results.txt", 'r')
# results = result_file.read().split("\n")

counter = 0
for i, filename in enumerate(images):
    # print('%s: %s' % (filename, labels[i]))
    img = imread('%s/%s' % (testing_dir, filename))

    # Detect license plate in image
    plate_segment = detect_plate(img)

    # Crop license plate in detected segment
    license_plate = segment_license_plate(plate_segment)

    # Segment characters in detected license plate frame
    characters = segment_characters(license_plate)

    if len(characters) == 7:
        print(''.join(classify_characters(characters)))
        # result_file.write(''.join(classify_characters(characters)))
    # else:
    #     print('')


# result_file.close()
label_file.close()