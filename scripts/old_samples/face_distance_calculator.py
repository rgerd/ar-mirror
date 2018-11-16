# A sample script for running haar cascades on
# the video feed from the built-in camera.

# Tutorials:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
# https://github.com/opencv/opencv/tree/master/data/haarcascades


## DETERMINED REASONABLE EYE DISTANCE @ 1D: 100px


import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

PIXELS_AT_STANDARD_DIST = 100.

face_dist = 1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        top_offset = int(h * 0.2)
        height_offset = int(h * 0.5)
        
        roi_gray = gray[y+top_offset:y+height_offset, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 2:
            left_eye  = eyes[0] if eyes[0][0] > eyes[1][0] else eyes[1]
            right_eye = eyes[0] if eyes[0][0] < eyes[1][0] else eyes[1]

            (lx, ly, lw, lh) = left_eye
            (rx, ry, rw, rh) = right_eye

            face_dist = 1 - (((lx + lw / 2) - (rx + rw / 2)) / PIXELS_AT_STANDARD_DIST)

        closeness = min(-min(0, face_dist * 2), 1)
        farness = min(max(0, face_dist * 2), 1)
        perfectness = 1 - min(np.square(face_dist) * 2, 1)

        cv.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (int(farness * 255),int(perfectness * 255),int(closeness * 255)),
            2)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()