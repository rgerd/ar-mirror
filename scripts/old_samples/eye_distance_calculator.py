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

left_eye = None
right_eye = None

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if frame is None or gray is None:
        continue

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ave_dist = 0
    num_samples = 0
    
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray = gray[y:y+int(h * 0.6), x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 2:
            left_eye  = eyes[0] if eyes[0][0] > eyes[1][0] else eyes[1]
            right_eye = eyes[0] if eyes[0][0] < eyes[1][0] else eyes[1]

            (lx, ly, lw, lh) = left_eye
            (rx, ry, rw, rh) = right_eye

            dist = (lx + lw / 2) - (rx + rw / 2)

            ave_dist += dist
            num_samples += 1
            print(float(ave_dist) / float(num_samples))

        if not (left_eye is None) and not (right_eye is None):
            (lx, ly, lw, lh) = left_eye
            (rx, ry, rw, rh) = right_eye

            cv.rectangle(roi_color, (lx,ly), (lx+lw,ly+lh), (255,0,0), 2)
            cv.rectangle(roi_color, (rx,ry), (rx+rw,ry+rh), (255,0,255), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()