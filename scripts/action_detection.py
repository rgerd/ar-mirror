import cv2 as cv
import numpy as np
import sys
from camera_reader import CameraReader
from keras.models import load_model
import matplotlib.pyplot as plt
import statistics

import collections

face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
model = load_model('keypoints_model.h5')

cv.startWindowThread()
cv.namedWindow('mirror')

camera = CameraReader()
camera.begin_reading()

px_hist = collections.deque([0], 20)
py_hist = collections.deque([0], 20)
vx_hist = collections.deque([0], 20)
vy_hist = collections.deque([0], 20)

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

    if len(faces) > 0:
        (x,y,w,h) = faces[0]

        # Add a red bounding box to the detections image
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)
        bgr_crop = frame[y:y+h, x:x+w]
        orig_shape_crop = bgr_crop.shape
        gray_crop = cv.cvtColor(bgr_crop, cv.COLOR_BGR2GRAY)
        resize_gray_crop = cv.resize(gray_crop, (96, 96)) / 255
        landmarks = np.squeeze(model.predict(
            np.expand_dims(np.expand_dims(resize_gray_crop, axis=-1), axis=0)))

        # x_offset = landmarks[20] - ((landmarks[22] + landmarks[24]) / 2.)
        x, y = landmarks[20], landmarks[21]
        vx = x - px_hist[-1]
        vy = y - py_hist[-1]
        px_hist.append(x); py_hist.append(y)
        vx_hist.append(vx); vy_hist.append(vy)

        stdv_x = np.std(vx_hist) * 100
        mean_x = np.mean(vx_hist) * 100
        stdv_y = np.std(vy_hist) * 100
        mean_y = np.mean(vy_hist) * 100
        print((stdv_y, mean_y))
        if stdv_x > 3 and abs(mean_x) < .5:
            print("NO")
            px_hist = collections.deque([0], 20)
            py_hist = collections.deque([0], 20)
            vx_hist = collections.deque([0], 20)
            vy_hist = collections.deque([0], 20)
        elif stdv_y > 2.5 and abs(mean_y) < .5:
            print("YES")
            px_hist = collections.deque([0], 20)
            py_hist = collections.deque([0], 20)
            vx_hist = collections.deque([0], 20)
            vy_hist = collections.deque([0], 20)

    # for i in range(0, len(landmarks), 2):
    #     xp = int((landmarks[i] * 48 + 48) * orig_shape_crop[0] / 96. + x)
    #     yp = int((landmarks[i+1] * 48 + 48) * orig_shape_crop[1] / 96. + y)
    #     cv.circle(frame, (xp, yp), 1, (0, 0, 255), 5)

    # Display the resulting frame
    cv.imshow('mirror', frame)

    # display plot
    # plt.gca().cla() # optionally clear axes
    # plt.plot(range(len(de)), de)
    # plt.title(str(i))
    # plt.draw()
    # plt.pause(0.1)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

plt.show(block=True)

# When everything is done, release the capture
camera.end_reading()
cv.destroyAllWindows()
