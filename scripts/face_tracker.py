import numpy as np
import cv2 as cv
from face import Face
import time
from camera_reader import CameraReader
import sys

face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

font = cv.FONT_HERSHEY_SIMPLEX

user = Face()

def update_face(image, min_size):
    updated_face_id = None # Array of indices

    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = min_size,
       )

    if len(faces) != 0:
        detected_face = max(faces, key=lambda f: f[2])
        (fx, fy, fw, fh) = detected_face
        face_img = image[fy:fy+fh,fx:fx+fw]

        user.observe((image.shape[1], image.shape[0]), detected_face, face_img)

def render_ui(user, screen):
    screen_width = screen.shape[1]
    screen_height = screen.shape[0]

    nx, ny = user.action_detector.get_position()
    cv.putText(screen, "%.2f, %.2f" % (nx, ny),
            (10, 40), cv.FONT_HERSHEY_SIMPLEX,
            2.0, user.get_settings().get_color(), 3)

    cv.line(screen,
        (int(screen_width / 2 - 50), int(screen_height / 2)), 
        (int(screen_width / 2 + 50), int(screen_height / 2)), 
        (255, 255, 255), 1)

    cv.line(screen,
        (int(screen_width / 2), int(screen_height / 2 - 50)), 
        (int(screen_width / 2), int(screen_height / 2 + 50)), 
        (255, 255, 255), 1)

    cv.circle(screen, 
        (int(nx + screen_width / 2), int(ny + screen_height / 2)), 
        5, (255, 255, 255), 2)

def main_loop(camera, mirror_mode):
    frame = camera.get_frame()
    
    if frame is None: # Wait for next frame
        return

    gray = frame
    frame = np.zeros((720, 1280, 3)) # Blank color image

    # Define min window size to be recognized as a face
    image_width, image_height = camera.get_img_dimensions()
    min_face_size = (int(0.05 * image_width), int(0.05 * image_height))
    update_face(gray, min_face_size)

    render_ui(user, frame)

    # Display the resulting frame
    # frame = cv.resize(frame, (0,0), fx=1.4, fy=1.4)
    cv.imshow('mirror', frame)

if __name__ == "__main__":
    mirror_mode = True # Just for black background

    cv.startWindowThread()
    cv.namedWindow('mirror')

    camera = CameraReader(mirror_mode)

    camera.begin_reading()
    while True:
        main_loop(camera, mirror_mode)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.end_reading()
    cv.destroyAllWindows()
