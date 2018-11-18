## DETERMINED REASONABLE EYE DISTANCE @ 1D: 100px

import numpy as np
import cv2 as cv
from face import Face

camera = cv.VideoCapture(0)
MIRROR_MODE = False

# Define min window size to be recognized as a face
minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('data/trainer/trainer.yml')

face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

font = cv.FONT_HERSHEY_SIMPLEX

face_dist = 1

names = ['Jared', 'Unknown', 'Kristy']
users = [ Face() ] * len(names) # One for each user

while(True):
    # Capture frame-by-frame
    ret, frame = camera.read()
    frame = cv.flip(frame, +1)
    background = frame
    if MIRROR_MODE:
        background = np.zeros(frame.shape)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for detected_face in faces:
        (fx, fy, fw, fh) = detected_face
        face_img = gray[fy:fy+fh,fx:fx+fw]

        id, confidence = recognizer.predict(face_img)

        if confidence >= 100: # Unknown face
            continue

        face = users[id]
        face.observe(detected_face, face_img)
        pos = face.kposition.value()

        # render face display
        color = face.get_distance_color()

        # observation
        cv.rectangle(
            background,
            (detected_face[0],detected_face[1]),
            (detected_face[0]+detected_face[2],detected_face[1]+detected_face[3]),
            (255, 255, 255),
            2)

        # kalman prediction
        cv.rectangle(
            background,
            (pos[0],pos[1]),
            (pos[0]+detected_face[2],pos[1]+detected_face[3]),
            color,
            2)

        cv.putText(background, str(names[id]), (fx+5,fy-5), font, 1, (255,255,255), 2)
        # cv.putText(frame, str(confidence), (fx+5,fy+fh-5), font, 1, (255,255,0), 1)

    # Display the resulting frame
    # background = cv.resize(background, (0,0), fx=1.4, fy=1.4)
    cv.imshow('window', background)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the camera
camera.release()
cv.destroyAllWindows()
