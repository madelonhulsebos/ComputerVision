# Imports
import numpy as np
import cv2

from os import walk

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 200
k = 2

positive_images = []
for (__, __, filename) in walk('../License plate detection/HOG/matches') :
    positive_images.extend(filename)
    break

for filename in positive_images[20:40] :

    img_original = cv2.imread('../License plate detection/HOG/matches/' + filename)
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_gauss_filtered = cv2.GaussianBlur(img_gray, (7, 7), 0)
    _, img_binary = cv2.threshold(img_gauss_filtered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    (img_contours, contours,_) = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    char_contours = []
    plate_contours = []
    for contour in contours :

        intX, intY, intWidth, intHeight = cv2.boundingRect(contour)

        if intWidth > 3*IMAGE_WIDTH/10 and intWidth < 8*IMAGE_WIDTH/10 and intHeight > IMAGE_HEIGHT/4 and intHeight < 2*IMAGE_HEIGHT/3 :
            img_lp = img_original[intY:intY+intHeight, intX:intX+intWidth,:]
            plate_contours.append(contour)

    if len(plate_contours) == 0 :
        invert_img = np.invert(img_binary)
        (img_contours, contours, _) = cv2.findContours(invert_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        plate_contours = []
        for contour in contours :

            intX, intY, intWidth, intHeight = cv2.boundingRect(contour)

            if intWidth > 3*IMAGE_WIDTH/10 and intWidth < 8*IMAGE_WIDTH/10 and intHeight > IMAGE_HEIGHT/4 and intHeight < 2*IMAGE_HEIGHT/3:
                img_lp = img_original[intY:intY+intHeight, intX:intX+intWidth,:]
                plate_contours.append(contour)

        if len(plate_contours) == 0 :

            (img_contours, contours, _) = cv2.findContours(img_gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

            # Customize character retrieval for this scenario: other thresholds needed for charactersizes etc.

            plate_contours = []
            for contour in contours :
                intX, intY, intWidth, intHeight = cv2.boundingRect(contour)
                img_lp = img_original[intY:intY+intHeight,intX:intX+intWidth,:]
                plate_contours.append(contour)

    img_lp = cv2.resize(img_lp, (150, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    img_gauss_filtered_lp = cv2.GaussianBlur(img_gray_lp, (1, 1), 0)
    ret, img_binary_lp = cv2.threshold(img_gray_lp, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (5, 5))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3,] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,147:150] = 255

    cv2.imshow("Binary license plate", img_binary_lp)

    (img_contours, contours, hierarchy) = cv2.findContours(img_binary_lp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    char_contours = []
    for contour in contours:

        intX, intY, intWidth, intHeight = cv2.boundingRect(contour)
        if intWidth < LP_WIDTH/2 and intHeight > 1*LP_HEIGHT/6 : #LP_WIDTH/20 and intWidth < 4*LP_WIDTH/7 and intHeight > 2*LP_HEIGHT/10 and intHeight < 9*LP_HEIGHT/10:
            char_contours.append(contour)
            cv2.rectangle(img_lp, (intX, intY), (intX + intWidth, intY + intHeight), (0, 255, 0), 1)

    if len(char_contours) == 7 :
        i = 0
        for contour in char_contours :
            filename = str(i)
            i = i + 1
            x,y,w,h = cv2.boundingRect(contour)
            char_img = img_lp[x-1:x+w+1,y-1:y+h+1]
            cv2.imwrite("../segmentation/Segmented_chars/" + filename, char_img)

    cv2.drawContours(img_lp, char_contours, -1, (0, 255, 0), 1)
    cv2.imshow("License plate", img_lp)
    cv2.waitKey(0)





