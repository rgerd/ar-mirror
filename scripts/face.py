import numpy as np
import cv2 as cv
from position import Position, KalmanPosition
import time
import datetime

# import matplotlib.pyplot as plt
from keras.models import load_model

PIXELS_AT_STANDARD_DIST = 1.
eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')
model = load_model('keypoints_model.h5')

class Face:
    def __init__(self):
        self.bounding_rect = None
        self.measured_position = Position(0, 0, 0)
        self.measured_size = (0, 0)
        self.filtered_position = KalmanPosition()

    def observe(self, frame_dim, face_rect, face_img):
        self.bounding_rect = face_rect

        frame_width, frame_height = frame_dim
        (fx, fy, fw, fh) = face_rect

        # EYE CASCADE METHOD
        def cascade():
            top_offset = int(fh * 0.2)
            height_offset = int(fh * 0.5)

            cropped_frame = face_img[top_offset:height_offset, :]

            eyes = eye_cascade.detectMultiScale(cropped_frame)

            z = None
            if len(eyes) == 2:
                left_eye  = eyes[0] if eyes[0][0] > eyes[1][0] else eyes[1]
                right_eye = eyes[0] if eyes[0][0] < eyes[1][0] else eyes[1]

                (lx, ly, lw, lh) = left_eye
                (rx, ry, rw, rh) = right_eye

                z = 1 - (((lx + lw / 2) - (rx + rw / 2)) / PIXELS_AT_STANDARD_DIST)
            return z # z = 0 @ when user is at standard distance

        # TRAINED KEY FEATURE METHOD
        def key_feature():
            resize_gray_crop = cv.resize(face_img, (96, 96)) / 255
            landmarks = np.squeeze(model.predict(
                np.expand_dims(np.expand_dims(resize_gray_crop, axis=-1), axis=0)))
            (rx, ry, lx, ly) = landmarks[0:4]
            
            # 500 is around the closest I could get to the mirror before it stopped detecting my face
            # 100 just felt like a good round number
            # Multiply by width since rx and lx come from a scaled image
            z = (500 - (rx - lx) * fw) / 100

            # Center x, y of face
            _cx = (rx + lx) / 2
            _cy = (ry + ly) / 2
            cx = int((_cx * 48 + 48) * fw / 96. + fx)
            cy = int((_cy * 48 + 48) * fh / 96. + fy)

            # 300 is to make a movement along the x/y axes feel similar to a movement along the z axis
            # Dividing by 100 just didn't seem like enough on these values (everything is arbitary)
            x =  (cx - frame_width / 2) / 300 * z
            y = -(cy - frame_height / 2) / 300 * z

            return Position(x, y, z)

        self.measured_position = key_feature()
        self.measured_size = (int(fw), int(fh))
        self.filtered_position.observe(time.time(), self.measured_position)

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
