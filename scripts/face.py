import numpy as np
import cv2 as cv
from position import Position, KalmanPosition
import time

PIXELS_AT_STANDARD_DIST = 100.
eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')

class Face:
    def __init__(self):
        self.bounding_rect = None
        self.measured_position = Position(0, 0, 0)
        self.filtered_position = KalmanPosition()

    def observe(self, face, face_img):
        self.bounding_rect = face

        (fx, fy, fw, fh) = face
        self.measured_position.x = fx
        self.measured_position.y = fy

        top_offset = int(fh * 0.2)
        height_offset = int(fh * 0.5)

        cropped_frame = face_img[top_offset:height_offset, :]

        eyes = eye_cascade.detectMultiScale(cropped_frame)

        if len(eyes) == 2:
          left_eye  = eyes[0] if eyes[0][0] > eyes[1][0] else eyes[1]
          right_eye = eyes[0] if eyes[0][0] < eyes[1][0] else eyes[1]

          (lx, ly, lw, lh) = left_eye
          (rx, ry, rw, rh) = right_eye

          z = 1 - (((lx + lw / 2) - (rx + rw / 2)) / PIXELS_AT_STANDARD_DIST)
          self.measured_position.z = z # z = 0 @ when user is at standard distance

        self.filtered_position.observe(time.time(), self.measured_position)

    def get_distance_color(self):
        (face_x, face_y, face_dist) = self.filtered_position.value()

        closeness = min(-min(0, face_dist * 2), 1)
        farness = min(max(0, face_dist * 2), 1)
        perfectness = 1 - min(np.square(face_dist) * 2, 1)

        return (int(farness * 255),int(perfectness * 255),int(closeness * 255))