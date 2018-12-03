import sys
import numpy as np
import cv2 as cv
from camera_reader import CameraReader
from ui_render import render_ui
from ar_render import render_ar
from face import Face

RENDER_SIZE = (540, 765)
SCREEN_SIZE = (1080, 1530)

WIDTH, HEIGHT = CameraReader(True).get_img_dimensions()
MIN_FACE_SIZE = (int(0.05 * WIDTH), int(0.05 * HEIGHT))
PPI = 57

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('data/trainer/trainer.yml')
face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

users = [Face('Jared', 0),
         Face('Robert', 0),
         Face('Unknown', 0)]

user_color = (0, 0, 0)
current_user = None
current_user_id = None
frames_on_new_user = 0

def main_loop(camera):
    global users, current_user_id, current_user, frames_on_new_user

    frame = camera.get_frame()
    if frame is None: # Wait for next frame
        return

    screen = np.zeros((RENDER_SIZE[1], RENDER_SIZE[0], 3), dtype=np.uint8)
    # cv.rectangle(screen, (0, 0), (RENDER_SIZE[0] - 2, RENDER_SIZE[1] - 2), (255, 255, 255), 3)

    user_id = update_faces(frame, MIN_FACE_SIZE)
    
    if user_id is not None:
        if user_id == current_user_id:
            frames_on_new_user = 0
        else:
            frames_on_new_user += 1

        if frames_on_new_user >= 8:
            current_user_id = user_id
            current_user = users[user_id]
            frames_on_new_user = 0
        
    if current_user_id is not None:
        render_ar(screen, PPI, current_user)    
        render_ui(screen, current_user.get_color())

    # display
    screen = cv.resize(screen, SCREEN_SIZE, interpolation=0)
    cv.imshow('image', screen)

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

        users[0].observe((image.shape[1], image.shape[0]), detected_face, face_img)

    return 0

def update_faces(image, min_size):
    updated_face_id = None # Array of indices

    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = min_size,
       )

    if len(faces) == 0: 
        return None

    fattest_face_id = None
    fattest_face_fatness = float('-inf')
    face_index = 0
    for frame in faces:
        (fx, fy, fw, fh) = frame
        face_img = image[fy:fy+fh,fx:fx+fw]

        id, _ = recognizer.predict(face_img)

        if fw >= fattest_face_fatness:
            fattest_face_id = id
            fattest_face_fatness = fw
        detected_user = users[id]
        detected_user.observe((image.shape[1], image.shape[0]), frame, face_img)
        updated_face_id = id
        
    return fattest_face_id

if __name__ == "__main__":
    local_mode = '--local' in sys.argv

    cv.startWindowThread()
    cv.namedWindow('image')
    cv.setWindowProperty('image', 0, 1)
    cv.moveWindow('image', 0, 0 if local_mode else -SCREEN_SIZE[1])

    camera = CameraReader(True)

    camera.begin_reading()

    while True:
        main_loop(camera)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.end_reading()
    cv.destroyAllWindows()
