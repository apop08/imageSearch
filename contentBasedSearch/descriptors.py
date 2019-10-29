# import the necessary packages
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model


def load_headless_pretrained_model():
    """
    Loads the pretrained resnet50 model with classification layer cut off
    """
    print("Loading headless pretrained model...")
    pretrained_resnet = ResNet50(weights='imagenet', include_top=False)
    return pretrained_resnet


class ResNetDescriptor:
    def __init__(self):
        self.model = load_headless_pretrained_model()

    def describe(self, image):
        return self.model.predict(image)


class HSVDescriptor:
    def __init__(self, bins):
        # number of bins to use in the histogram
        self.bins = bins

    def describe(self, image):
        """Creates masks and describes the histogram of
        each region

        Parameters:
            image (image): image to describe

        Returns:
            np.array: returns feature vector
        """
        # convert to hsv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # create features array
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                    (0, cX, cY, h)]

        # construct an elliptical mask representing the center
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        # return the feature vector
        return np.array(features)

    def histogram(self, image, mask=None):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel; then
        # normalize the histogram
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])

        # otherwise handle for OpenCV 3+
        hist = cv2.normalize(hist, hist).flatten()

        # return the histogram
        return hist
