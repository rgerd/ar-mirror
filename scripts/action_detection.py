import cv2
import numpy as np
import sys

from keras.models import load_model

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
model = load_model('keypoints_model.h5')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    min_size = (int(0.05 * frame.shape[1]), int(0.05 * frame.shape[0]))
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = min_size
       )

    for (x,y,w,h) in faces:
        # Add a red bounding box to the detections image
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)
        bgr_crop = frame[y:y+h, x:x+w]
        orig_shape_crop = bgr_crop.shape
        gray_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        resize_gray_crop = cv2.resize(gray_crop, (96, 96)) / 255
        landmarks = np.squeeze(model.predict(
            np.expand_dims(np.expand_dims(resize_gray_crop, axis=-1), axis=0)))


        for i in range(0, len(landmarks), 2):
            xp = int((landmarks[i] * 48 + 48) * orig_shape_crop[0] / 96. + x)
            yp = int((landmarks[i+1] * 48 + 48) * orig_shape_crop[1] / 96. + y)
            cv2.circle(frame, (xp, yp), 2, (0, 0, 255), 5)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
