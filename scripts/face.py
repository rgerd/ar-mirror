import numpy as np
import cv2 as cv
from position import Position
from action_detection import ActionDetector
from keras.models import load_model

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 255, 255)
]

keypoints_model = load_model('keypoints_model.h5')

class Face:
    def __init__(self, name, color_index):
        self.name = name
        self.color_index = color_index
        self.bounding_rect = None
        self.measured_position = Position(0, 0, 0)
        self.predicted_position = Position(0, 0, 0)
        self.measured_size = (0, 0)
        self.nose = (0, 0)
        self.action_detector = ActionDetector(self._do_action)

    def _get_nose(self, face_img):
        resize_gray_crop = cv.resize(face_img, (96, 96)) / 255
        landmarks = np.squeeze(keypoints_model.predict(np.expand_dims(np.expand_dims(resize_gray_crop, axis=-1), axis=0)))
        return (landmarks[20], landmarks[21])

    def _predict_position(self, frame_dim, face_rect):
        frame_width, frame_height = frame_dim
        fx, fy, fw, fh = face_rect

        # Normalized face size, independant of frame size scaling
        # Faces are typically taller than they are wide, so we'll probably hit fh > frame_height first
        face_size = fh / frame_height

        z = (1.0 - face_size) * 32

        # Average head width: my estimation 5.5, internet: 7.1
        cam_focal_length = 300.

        # Center x, y of face
        cx = fx + fw / 2
        cy = fy + fh / 3

        x =  (cx - frame_width / 2) * z / cam_focal_length
        y = -(cy - frame_height / 2) * z / cam_focal_length

        x = round(x, 2); y = round(y, 2); z = round(z, 2)

        current_position = Position(x, y, z)

        predicted_position = self.measured_position.update(current_position, 0.7)

        self.measured_position = current_position

        return predicted_position

    def observe(self, frame_dim, face_rect, face_img):
        self.bounding_rect = face_rect
        self.nose = self._get_nose(face_img)
        self.predicted_position = self._predict_position(frame_dim, face_rect)
        self.measured_size = (int(face_rect[2]), int(face_rect[3]))
        self.action_detector.update(self.nose)

    def _do_action(self, action):
        if action == "nod":
            color_increment = 1
        elif action == "shake":
            color_increment = -1
        self.color_index += color_increment
        if self.color_index < 0:
            self.color_index += len(colors)
        self.color_index %= len(colors)

    def get_color(self):
        return colors[self.color_index]
