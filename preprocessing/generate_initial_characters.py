from skimage import data, filters, io, img_as_uint
from os import walk
import re

license_dir = '../datasets/characters/licenses'
character_base_dir = '../datasets/characters/base'

_, _, license_files = next(walk(license_dir))

for filename in license_files:
    characters = list(re.sub('\.jpg$', '', filename))
    license = io.imread('%s/%s' % (license_dir, filename), as_grey=True)
    character_strip = license[36:80, 16:184]

    for i, character in enumerate(characters):
        start = int(i * character_strip.shape[1] / len(characters))
        end = int(start + character_strip.shape[1] / len(characters))

        image = character_strip[:, start:end]
        thresh = filters.threshold_otsu(image)
        binary = image < thresh

        io.imsave(fname='%s/%s.bmp' % (character_base_dir, character),
                  arr=img_as_uint(binary))



