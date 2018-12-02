# https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

import numpy as np
import cv2 as cv
import imutils
from imutils import paths
 
from camera_reader import CameraReader

def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(gray, 35, 125)
 
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = max(cnts, key = cv.contourArea)
 
    # compute the bounding box of the of the paper region and return it
    return cv.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 44.0
 
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0

# 225, 300
def main_loop(camera):
    image = camera.get_frame()
    
    if image is None: # Wait for next image
        return
 
    # load the first image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length
    marker = find_marker(image)

    focalLength = 300
    if focalLength is None:
        focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
        print(focalLength)
    else:
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
        cv.putText(image, "%.2fft" % (inches / 12),
            (image.shape[1] - 200, image.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 0), 3)

    # draw a bounding box around the image and display it
    box = cv.cv.BoxPoints(marker) if imutils.is_cv2() else cv.boxPoints(marker)
    box = np.int0(box)
    cv.drawContours(image, [box], -1, (0, 255, 0), 2)

    cv.imshow('image', image)

if __name__ == "__main__":
    cv.startWindowThread()
    cv.namedWindow('image')

    camera = CameraReader()

    camera.begin_reading()

    while True:
        main_loop(camera)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.end_reading()
    cv.destroyAllWindows()


