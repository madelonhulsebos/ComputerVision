from skimage import io, filters, img_as_uint, morphology, transform, util
from os import walk
import warnings
import random
import uuid
import re

warnings.filterwarnings("ignore")


def generate_initial_characters():
    base_characters_dir = '../datasets/characters/base'

    # Split the raw license plates into individual characters
    raw_licenses_dir = '../datasets/characters/raw/licenses'
    _, _, license_files = next(walk(raw_licenses_dir))
    for filename in license_files:
        characters = list(re.sub('\.jpg$', '', filename))
        license = io.imread('%s/%s' % (raw_licenses_dir, filename), as_grey=True)
        character_strip = license[36:80, 16:184]

        for i, character in enumerate(characters):
            start = int(i * character_strip.shape[1] / len(characters))
            end = int(start + character_strip.shape[1] / len(characters))

            image = character_strip[:, start:end]
            thresh = filters.threshold_otsu(image)
            binary = image < thresh

            io.imsave(fname='%s/%s_1.bmp' % (base_characters_dir, character),
                      arr=img_as_uint(binary))

    # Split the images containing different raw characters
    raw_characters_dir = '../datasets/characters/raw/characters'
    filename = '1234567890.png'
    characters = list(re.sub('\.png$', '', filename))
    character_map = io.imread('%s/%s' % (raw_characters_dir, filename), as_grey=True)
    character_strip = character_map[:, 5:-9]

    for i, character in enumerate(characters):
        start = int(i * character_strip.shape[1] / len(characters))
        end = int(start + character_strip.shape[1] / len(characters))

        image = character_strip[:, start:end]
        image = transform.resize(image, (44, 21))

        thresh = filters.threshold_otsu(image)
        binary = image < thresh
        binary = util.pad(binary,
                          pad_width=((0, 0), (1, 2)),
                          mode='constant',
                          constant_values=0)

        io.imsave(fname='%s/%s_2.bmp' % (base_characters_dir, character),
                  arr=img_as_uint(binary))


def generate_noisy_characters(train, amount=None, dropout_rate=0.02, morph_rate=0.4, resize_rate=0.4,
                              rescale_rate=0.4, width=24, height=44):

    save_dir = '../datasets/characters/noisy_%s' % ('train' if train is True else 'test')
    base_characters_dir = '../datasets/characters/base'

    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    c = 1
    limit = float('Inf') if amount is None else int(amount)
    while c < limit:
        character = random.choice(characters)
        image = io.imread('%s/%s_%d.bmp' % (base_characters_dir, character, random.randint(1, 2)), as_grey=True)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Go over each pixel and if it is a positive one, implement a chance of deleting it.
                if random.random() < dropout_rate:
                    image[i][j] = 0

        if random.random() < morph_rate:
            image = morphology.erosion(image, morphology.square(random.randint(1, 3)))

        if random.random() < rescale_rate:
            shrinkage_y = min(random.random(), 0.2)
            shrinkage_x = min(random.random(), 0.2)
            image = transform.rescale(image, scale=(1 - shrinkage_y, 1 - shrinkage_x))

        if random.random() < resize_rate:
            shrinkage_y = int(random.randint(1, int(image.shape[0] / 20)) * 4)
            shrinkage_x = int(shrinkage_y / 2)
            image = transform.resize(image, (image.shape[0] - shrinkage_y, image.shape[1] - shrinkage_x))

        # Perform a random (small) rotation
        image = transform.rotate(image, random.randint(-10, 10) * random.random())

        # Make sure the final image has the correct dimensions
        pad_width_y = int((height - image.shape[0]) / 2)
        pad_width_x = int((width - image.shape[1]) / 2)
        image = util.pad(image,
                         pad_width=((pad_width_y, pad_width_y), (pad_width_x, pad_width_x)),
                         mode='constant',
                         constant_values=0)
        image = transform.resize(image, (height, width))

        io.imsave(fname='%s/%s_%s.jpg' % (
        save_dir, characters.index(character), re.sub(r"-", '', str(uuid.uuid4()))),
                  arr=img_as_uint(image))

        if c % 1000 == 0:
            print('Saved %d noisy characters!' % c)

        c += 1