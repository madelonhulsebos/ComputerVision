import numpy as np
import cv2

# Match contours to license plate or character template
def _check_contours(boundaries, img_orig, img_preproc, license_plate_check) :

    # Find all contours in the image
    (_, cntrs, _) = cv2.findContours(img_preproc.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential boundaries
    lower_width = boundaries[0]
    upper_width = boundaries[1]
    lower_height = boundaries[2]
    upper_height = boundaries[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    if license_plate_check is True :
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:5]
    else :
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :

        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        # Check if contour has proper sizes
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :

            x_cntr_list.append(intX)
            target_contours.append(cntr)

            # If we check a license plate, crop the license plate
            if license_plate_check is True :
                img_res = img_orig[intY:intY+intHeight, intX:intX+intWidth, :]

            # If we check a character, crop the character
            if license_plate_check is False :

                char_copy = np.zeros((44,24))
                char = img_orig[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy)

    #Return characters based on
    if license_plate_check is not True:
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        target_contours_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])
            target_contours_copy.append(target_contours[idx])
        img_res = img_res_copy
        target_contours = target_contours_copy

    return target_contours, img_res

# Crop license plates
def segment_license_plate(image) :

    IMAGE_HEIGHT = 100
    IMAGE_WIDTH = 200

    # Preprocess image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Define expected boundaries of detected license plate
    lower_width_lp = IMAGE_WIDTH/4
    upper_width_lp = 8*IMAGE_WIDTH/10
    lower_height_lp= IMAGE_HEIGHT/4
    upper_height_lp = 2*IMAGE_HEIGHT/3

    boundaries_lp = [lower_width_lp,
                  upper_width_lp,
                  lower_height_lp,
                  upper_height_lp]


    # Retrieve the probable cropped license plate
    plate_contours, img_lp = _check_contours(boundaries_lp, image, img_binary, True)

    if len(plate_contours) == 0 :
        invert_img = np.invert(img_binary)

        # Check contour of inverted image if it's possibly a license plate
        plate_contours, img_lp = _check_contours(boundaries_lp, image, invert_img, True)


    # If no license plate was found, return the biggest contour
    if len(plate_contours) == 0 :

         (_, contours, _) = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

         plate_contours = []
         for contour in contours :
             intX, intY, intWidth, intHeight = cv2.boundingRect(contour)
             img_lp = image[intY:intY+intHeight,intX:intX+intWidth,:]
             plate_contours.append(contour)

    return img_lp



# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (150, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,147:150] = 255

    # Estimations of character contours sizes of cropped license plates
    boundaries_crop = [LP_WIDTH/6,
                       LP_WIDTH/3,
                       LP_HEIGHT/6,
                       2*LP_HEIGHT/3]

    # Estimations of character contour sizes of non-cropped license plates
    boundaries_no_crop = [LP_WIDTH/12,
                          LP_WIDTH/6,
                          LP_HEIGHT/8,
                          LP_HEIGHT/3]

    # Get contours within cropped license plate
    char_contours, char_list = _check_contours(boundaries_crop, img_binary_lp, img_binary_lp, False)

    if len(char_contours) != 7:

        # Check the smaller contours; possibly no plate was detected at all
        char_contours, char_list = _check_contours(boundaries_no_crop, img_binary_lp, img_binary_lp, False)


    if len(char_contours) == 0 :

            # If nothing was found, try inverting the image in case the background is darker than the foreground
            invert_img_lp = np.invert(img_binary_lp)
            char_contours, char_list = _check_contours(boundaries_crop, img_binary_lp, invert_img_lp, False)

    # If we found 7 chars, it is likely to form a license plate
    full_license_plate = []
    if len(char_contours) == 7 :

        full_license_plate = char_list

    return full_license_plate