## DETERMINED REASONABLE EYE DISTANCE @ 1D: 100px
import numpy as np
import cv2 as cv
from face import Face
import time
from camera_reader import CameraReader

MIRROR_MODE = False

camera = CameraReader()

window_width, window_height = camera.get_img_dimensions()
# Define min window size to be recognized as a face
minW = 0.05 * window_width
minH = 0.05 * window_height

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('data/trainer/trainer.yml')

face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

font = cv.FONT_HERSHEY_SIMPLEX

names = ['Jared', 'Kristy', 'Unknown']
user = Face()

# axis centered at the center of the mirror.
# positive x to the right. y is up. z is out towards user.
def pixel_to_center_frame(point):
    point[0] -= window_width / 2
    point[1] -= window_height / 2
    return point

def center_frame_to_pixel(point):
    point[0] += window_width / 2
    point[1] += window_height / 2
    return point

def mirror_point(perspective, location):
    if perspective[2] <= 0:
        print('invalid perspective: face is behind the mirror')
        return (0,0)
    if location[2] > 0:
        print('invalid location: positive z position for location... flipping sign')
        location[2] = -location[2]

    # calculate for z = 0
    (px, py, pz) = perspective
    (lx, ly, lz) = location
    x = px - ((px - lx) / float(pz - lz)) * pz
    y = py - ((py - ly) / float(pz - lz)) * pz

    return center_frame_to_pixel([x, y])

def main_loop():
    gray = camera.get_frame()
    if gray is None:
        return

    frame = gray
    if MIRROR_MODE:
        frame = np.zeros(gray.shape)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    if len(faces) != 0:
        detected_face = max(faces, key=lambda f: f[2])
        (fx, fy, fw, fh) = detected_face
        face_img = gray[fy:fy+fh,fx:fx+fw]

        id, confidence = recognizer.predict(face_img)

        if confidence >= 100: # Unknown face
           return

        user.observe(detected_face, face_img)

        # render face display
        color = user.get_distance_color()
        filtered_color = user.get_distance_color(filter=True)

        # calculate perspective (assuming camera is center of mirror)
        (x, y, z) = user.filtered_position.value()
        # normalize
        x += fw / 2.
        y += fw / 3.
        cv.circle(frame, (int(x), int(y)), 5, filtered_color, 2)

        point = [0, 0, -2] # center of mirror with depth of 10
        perspective = pixel_to_center_frame([x,y,z])
        static_x, static_y = mirror_point(perspective, point)
        cv.circle(frame, (int(static_x), int(static_y)), 5, (255, 255, 255), 2)

        cv.putText(frame, str(names[id]), (fx+5,fy-5), font, 1, (255,255,255), 2)

    # Display the resulting frame
    # frame = cv.resize(frame, (0,0), fx=1.4, fy=1.4)
    cv.imshow('window', frame)


if __name__ == "__main__":
    camera.begin_reading()
    while True:
        main_loop()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.end_reading()
    cv.destroyAllWindows()
