import numpy as np
import cv2 as cv
from position import Position, KalmanPosition
from action_detection import ActionDetector
import time
import datetime

# import matplotlib.pyplot as plt
from keras.models import load_model

PIXELS_AT_STANDARD_DIST = 1.
# eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')

class Face:
    def __init__(self):
        self.bounding_rect = None
        self.measured_position = Position(0, 0, 0)
        self.measured_size = (0, 0)
        self.filtered_position = KalmanPosition()
        self.nose = (0, 0)
        self.keypoints_model = load_model('keypoints_model.h5')
        self.action_detector = ActionDetector(self._do_action)

    def _get_nose(self, face_img):
        resize_gray_crop = cv.resize(face_img, (96, 96)) / 255
        landmarks = np.squeeze(self.keypoints_model.predict(np.expand_dims(np.expand_dims(resize_gray_crop, axis=-1), axis=0)))
        return (landmarks[20], landmarks[21])

    def _predict_position(self, frame_dim, face_rect):
        frame_width, frame_height = frame_dim
        fx, fy, fw, fh = face_rect

        # Normalized face size, independant of frame size scaling
        # Faces are typically taller than they are wide, so we'll probably hit fh > frame_height first
        face_size = fh / frame_height

        z = (1.0 - face_size) * 32
    
        # my estimation 5.5, internet: 7.1
        cam_focal_length = 300.

        # Center x, y of face
        cx = fx + fw / 2
        cy = fy + fh / 2

        x =  (cx - frame_width / 2) * z / cam_focal_length
        y = -(cy - frame_height / 2) * z / cam_focal_length

        x = round(x, 2); y = round(y, 2); z = round(z, 2)

        current_position = Position(x, y, z)

        predicted_position = self.measured_position.update(current_position, 0.7)

        self.measured_position = current_position

        # self.filtered_position.observe(time.time(), self.measured_position)
        return predicted_position

    def observe(self, frame_dim, face_rect, face_img):
        self.bounding_rect = face_rect
        self.nose = self._get_nose(face_img)
        self.predicted_position = self._predict_position(frame_dim, face_rect)
        self.measured_size = (int(face_rect[2]), int(face_rect[3]))
        self.action_detector.update(self.nose)

    def get_distance_color(self, filter=False):
        if filter:
            (_, _, face_dist) = self.filtered_position.value()
        else:
            face_dist = self.measured_position.z

        face_dist = face_dist - 3 # Pretty standard distance

        closeness = min(-min(0, face_dist * 2), 1)
        farness = min(max(0, face_dist * 2), 1)
        perfectness = 1 - min(np.square(face_dist) * 2, 1)

        return (int(farness * 255),int(perfectness * 255),int(closeness * 255))

    def _do_action(self, action):
        if action == "nod":
            print("OH YEAH")
        elif action == "shake":
            print("OH NO")
