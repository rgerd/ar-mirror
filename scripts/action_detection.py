import cv2 as cv
import numpy as np
import sys
from camera_reader import CameraReader

from keras.models import load_model

face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
model = load_model('keypoints_model.h5')

cv.startWindowThread()
cv.namedWindow('mirror')

camera = CameraReader()

camera.begin_reading()

while True:
    # Capture frame-by-frame
    frame = camera.get_frame()
    
    if frame is None: # Wait for next frame
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    min_size = (int(0.05 * frame.shape[1]), int(0.05 * frame.shape[0]))
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = min_size
       )

    for (x,y,w,h) in faces:
        # Add a red bounding box to the detections image
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)
        bgr_crop = frame[y:y+h, x:x+w]
        orig_shape_crop = bgr_crop.shape
        gray_crop = cv.cvtColor(bgr_crop, cv.COLOR_BGR2GRAY)
        resize_gray_crop = cv.resize(gray_crop, (96, 96)) / 255
        landmarks = np.squeeze(model.predict(
            np.expand_dims(np.expand_dims(resize_gray_crop, axis=-1), axis=0)))


        for i in range(0, len(landmarks), 2):
            xp = int((landmarks[i] * 48 + 48) * orig_shape_crop[0] / 96. + x)
            yp = int((landmarks[i+1] * 48 + 48) * orig_shape_crop[1] / 96. + y)
            cv.circle(frame, (xp, yp), 2, (0, 0, 255), 5)

    # Display the resulting frame
    cv.imshow('mirror', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
camera.end_reading()
cv.destroyAllWindows()
