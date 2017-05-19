# ComputerVision

For our Computer Vision Project we would like to build a license plate recognizer. Our goal
is to build a system that takes an image as input, checks whether it contains a license plate, and
if it does, identifies the characters from that license plate. In order to achieve this, we believe our
system should be capable of performing the following tasks:
1. License plate detection: Identify which part of the image, if any, depicts the license plate
and extract it. For this we will look into applications of corner detection and state-of-the-art
object recognition techniques.
2. Object transformation: Apply the necessary transformations to the extracted license plate
segment so that it has a standardized, rectangular shape.
3. Component detection: Use component detection algorithms to extract the individual
characters from the license plate as separate segments. We plan to use standard connectedcomponent
labeling algorithms, but it may be that more sophisticated algorithms are required.
4. Feature extraction: Extract relevant features from each separate character segment for
classification. We are currently thinking of using SIFT and/or HOG features for this purpose.
5. Optical character recognition: Classify each character segment to identify which character
(A-Z, 0-9, or a dash (-)) is depicted. We would like to use a neural network to do this.
Once the system has classified each individual character we can concatenate all the character labels
to get the desired output. Figure 1 shows the pipeline of our system as we currently envision it.

![System pipeline](./images/pipeline.jpg)
*Figure 1: Overview of our system's pipeline*

In order to train the neural network that we will be using for the classification task we will first
try and find readily available datasets online. If we are unable to find suitable datasets we plan to
manually build a small training set ourselves. We can then either apply semi-supervised learning
techniques alongside other unlabeled samples or generate additional synthetic samples (by changing
the background or adding noise) to enlarge this training set.

To evaluate our system as a whole we plan on generating a test set by taking several photos
of cars ourselves, which we could then combine with existing photos from online datasets. We can
then simply measure the system’s accuracy on this test set. Additionally, we can vary different
properties of these photos (such as scale, point of view, brightness, weather conditions, etc.) to see
how robust our system is to such variations.

As for the implementation details, we would like to use Python as our language of choice. For
the image processing and machine learning tasks we will make use of scikit-image and Tensorflow
or scikit-learn, respectively.
