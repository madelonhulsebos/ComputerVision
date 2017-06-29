from os import walk
from skimage.io import imread

from detection.HOG.detect_license_plate import detect_plate
from segmentation.Segmentation import segment_license_plate, segment_characters
from recognition.cnn.test import classify_characters

testing_dir = 'datasets/cars_markus'

for (_, _, images) in walk(testing_dir):
    break

result_file = open("results.txt", "w")
label_file = open("labels.txt", "r")
labels = label_file.read().split("\n")

for i, filename in enumerate(images):
    print('Analyzing file %s with license plate %s' % (filename, labels[i]))
    img = imread('%s/%s' % (testing_dir, filename))

    # Detect license plate in image
    plate_segment = detect_plate(img)

    # Crop license plate in detected segment
    license_plate = segment_license_plate(plate_segment)

    # Segment characters in detected license plate frame
    characters = segment_characters(license_plate)

    if len(characters) == 7:
        print('Found a complete license plate: %s' % ''.join(classify_characters(characters)))
        result_file.write(''.join(classify_characters(characters)))
    else:
        print('Unable to find all 7 characters!')

    result_file.write("\n")

result_file.close()
label_file.close()
