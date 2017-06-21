from skimage import morphology, data, filters, io, img_as_ubyte, transform, util, img_as_uint
from os import walk
import re
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import uuid
import warnings

warnings.filterwarnings("ignore")

IMG_WIDTH = 24
IMG_HEIGHT = 44

DROPOUT_RATE = 0.02
MORPH_RATE = 0.4
RESIZE_RATE = 0.4
RESCALE_RATE = 0.4

noisy_character_dir = '../datasets/characters/noisy'
base_characters_dir = '../datasets/characters/base'
characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

c = 1
while True:
    character = random.choice(characters)
    image = io.imread('%s/%s.bmp' % (base_characters_dir, character), as_grey=True)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Go over each pixel and if it is a positive one, implement a chance of deleting it.
            if random.random() < DROPOUT_RATE:
                image[i][j] = 0

    if random.random() < MORPH_RATE:
        image = morphology.erosion(image, morphology.square(random.randint(1, 3)))

    if random.random() < RESCALE_RATE:
        shrinkage_y = min(random.random(), 0.2)
        shrinkage_x = min(random.random(), 0.2)
        image = transform.rescale(image, scale=(1 - shrinkage_y, 1 - shrinkage_x))

    if random.random() < RESIZE_RATE:
        shrinkage_y = int(random.randint(1, int(image.shape[0] / 20)) * 4)
        shrinkage_x = int(shrinkage_y / 2)
        image = transform.resize(image, (image.shape[0] - shrinkage_y, image.shape[1] - shrinkage_x))

    # Perform a random (small) rotation
    image = transform.rotate(image, random.randint(-10, 10) * random.random())

    # Make sure the final image has the correct dimensions
    pad_width_y = int((IMG_HEIGHT - image.shape[0]) / 2)
    pad_width_x = int((IMG_WIDTH - image.shape[1]) / 2)
    image = util.pad(image,
                     pad_width=((pad_width_y, pad_width_y), (pad_width_x, pad_width_x)),
                     mode='constant',
                     constant_values=0)
    image = transform.resize(image, (IMG_HEIGHT, IMG_WIDTH))

    io.imsave(fname='%s/%s_%s.jpg' % (noisy_character_dir, character, re.sub(r"-", '', str(uuid.uuid4()))),
              arr=img_as_uint(image))

    if c % 1000 == 0:
        print('Just saved image #%d' % c)

    c += 1
