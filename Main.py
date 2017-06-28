from os import walk
from skimage.io import imread

from detection.HOG.detect_license_plate import detect_plate
from segmentation.Segmentation import segment_license_plate, segment_characters

testing_dir = 'datasets/cars_markus'

for (_, _, images) in walk(testing_dir):
    break

counter = 0
for filename in images :

    img = imread('%s/%s' % (testing_dir, filename))

    # Detect license plate in image
    plate_segment = detect_plate(img)

    # Crop license plate in detected segment
    license_plate = segment_license_plate(plate_segment)

    # Segment characters in detected license plate frame
    characters = segment_characters(license_plate)

    if len(characters) == 7 :

        print('call CNN to classify characters')