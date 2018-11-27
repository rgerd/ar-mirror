import numpy as np
import cv2 as cv
from face import Face
import time
from camera_reader import CameraReader
import sys

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('data/trainer/trainer.yml')

face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

font = cv.FONT_HERSHEY_SIMPLEX

names = ['Jared', 'Kristy', 'Unknown']
users = [Face(), Face(), Face()]

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

        id, confidence = recognizer.predict(face_img)

        if confidence >= 100: # Unknown face
           return updated_face_id

        detected_user = users[id]
        detected_user.observe((image.shape[1], image.shape[0]), detected_face, face_img)
        updated_face_id = id

    return updated_face_id

def render_ui(user, user_name, screen):
    screen_width = screen.shape[1]
    screen_height = screen.shape[0]

    # render face display
    color = user.get_distance_color()
    filtered_color = user.get_distance_color(filter=True)

    # calculate perspective (assuming camera is center of mirror)
    (x, y, z) = user.filtered_position.value()

    # The mirror
    cv.line(
        screen, 
        (int(screen_width / 2 - 40), int(screen_height / 2)), 
        (int(screen_width / 2 + 40), int(screen_height / 2)), 
        (255, 255, 255)
    )

    # Top-down view of the user's head
    cv.circle(screen, (int(x * 20 + screen_width / 2), int(z * 20 + screen_height / 2)), 5, (255, 255, 255), 2)

def main_loop(camera, mirror_mode):
    frame = camera.get_frame()
    
    if frame is None: # Wait for next frame
        return

    image_width, image_height = camera.get_img_dimensions()
    # Define min window size to be recognized as a face
    min_face_size = (int(0.05 * image_width), int(0.05 * image_height))

    if mirror_mode:
        gray = frame
        frame = np.zeros((image_height, image_width, 3)) # Blank color image
    else:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    user_id = update_face(gray, min_face_size)

    if user_id is None:
        return

    render_ui(users[user_id], names[user_id], frame)

    # Display the resulting frame
    # frame = cv.resize(frame, (0,0), fx=1.4, fy=1.4)
    cv.imshow('window', frame)

if __name__ == "__main__":
    mirror_mode = True # Just for black background

    camera = CameraReader(mirror_mode)

    camera.begin_reading()
    while True:
        main_loop(camera, mirror_mode)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.end_reading()
    cv.destroyAllWindows()
