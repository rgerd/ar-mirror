import numpy as np
import cv2 as cv

PIXELS_AT_STANDARD_DIST = 100.
eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Face:
    def __init__(self):
        self.bounding_rect = None
        self.position = Vector3(0, 0, 0)

    def observe_face(self, face, face_img):
        self.bounding_rect = face

        (fx, fy, fw, fh) = face

        top_offset = int(fh * 0.2)
        height_offset = int(fh * 0.5)
        
        cropped_frame = face_img[top_offset:height_offset, :]
        
        eyes = eye_cascade.detectMultiScale(cropped_frame)

        if len(eyes) == 2:
            left_eye  = eyes[0] if eyes[0][0] > eyes[1][0] else eyes[1]
            right_eye = eyes[0] if eyes[0][0] < eyes[1][0] else eyes[1]

            (lx, ly, lw, lh) = left_eye
            (rx, ry, rw, rh) = right_eye

            self.position.z = 1 - (((lx + lw / 2) - (rx + rw / 2)) / PIXELS_AT_STANDARD_DIST)


    def get_distance_color(self):
        face_dist = self.position.z

        closeness = min(-min(0, face_dist * 2), 1)
        farness = min(max(0, face_dist * 2), 1)
        perfectness = 1 - min(np.square(face_dist) * 2, 1)

        return (int(farness * 255),int(perfectness * 255),int(closeness * 255))



    # def detect_eyes(self, gray_face):        
    #     roi_gray = gray_face[y:y+int(self.bounding_rect[3] * 0.6), x:x+]
    #     roi_color = frame[y:y+h, x:x+w]
        
    #     new_eyes = eye_cascade.detectMultiScale(roi_gray)



    #     for (ex,ey,ew,eh) in eyes:
    #         cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
