# ComputerVision

For our Computer Vision Project we have built a license plate recognizer. The system takes
an image as input, checks whether it contains a license plate, and
if it does, identifies and classifies the characters from that license plate. 
To achieve this, three main components are integrated:
1. License plate detection: Identifies which part of the image, if any, depicts the license plate
and extracts it. To do so we implemented a variation of Dalal-Trigg's object detector, using a non-linear SVM instead of a linear one and HOG features to represent the input images.
2. Character segmentation: Identifies the individual characters in the detected segments. We first cropped each license plate segment to a license plate without background. The characters were extracted afterwards. Both subcomponents use contours of the objects in an image to identify whether it is a license plate or characters respectively.
3. Optical character recognition: Classify each character segment to identify which character
(A-Z, 0-9) is depicted. We implemented this component using a 3-layered Convolutional Neural Network.

Once the system has classified each individual character we concatenated all the character labels
to get the desired output. Figure 1 shows the resulting pipeline of this system.

![System pipeline](./images/ComputerVision_System.jpg)

*Figure 1: Overview of our system's pipeline*

### Instructions to execute system

1. Clone this repository 
2. Run the `Main.py` file (input images automatically imported)
3. Resulting classifications can be found in the `results.txt` file in the root (while running we also print progress updates in the console)

In `results.txt`, each line contains the predicted text for the license plate in the corresponding image (i.e. the first line contains the prediction of image_0001.jpg). The images that we test on are located in the `datasets/cars_markus` directory. For the correct labels of each license plate in these images, please see the `labels.txt` file (which follows the same structure). Here we used "IRREGULAR" to denote a license plate that was in some way ill-suited for our classification task (because part of the plate was obscured, or because it contained highly irregular license plates such as "NEW CENTURY").

### Overview of used packages and frameworks

Each system component is written in Pyton, where we used several packages as listed in table 1.


| Commponent             | Package       | 
| ---------------------- |:-------------:| 
| General                |Scikit-learn, scikit-image, numpy |
| Detection              | - | 
| Segmentation           | OpenCV   | 
| Charachter recognition | TensorFlow    | 

*Table 1: Overview of used packages and frameworks*

