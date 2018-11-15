# A sample script for running haar cascades on
# the video feed from the built-in camera.

# Tutorials:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
# https://github.com/opencv/opencv/tree/master/data/haarcascades

import numpy as np
import cv2 as cv
from face import Face

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

faces = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    possible_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(possible_faces) == 0:
        del faces[:]

    for face in faces:
        detected_face_index = face.update_face(gray, possible_faces)

        if detected_face_index != -1:
            del possible_faces[detected_face_index]

    for new_face in possible_faces:
        faces.append(Face())
        faces[-1].set_face(new_face)

    for updated_face in faces:
        (x, y, w, h) = updated_face.bounding_rect
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # roi_gray = gray[y:y+int(h * 0.6), x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        
        # new_eyes = eye_cascade.detectMultiScale(roi_gray)

        # for (ex,ey,ew,eh) in eyes:
        #     cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()