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

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('data/trainer/trainer.yml')
face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

names = ['Jared', 'Kristy', 'Unknown']
users = [Face(), Face(), Face()]

def main_loop(camera):
    frame = camera.get_frame()
    if frame is None: # Wait for next frame
        return

    user_id = update_face(frame, MIN_FACE_SIZE)
    if user_id is None:
        return

    # render
    screen = np.zeros((RENDER_SIZE[1], RENDER_SIZE[0], 3), dtype=np.uint8)
    cv.rectangle(screen, (0, 0), (RENDER_SIZE[0] - 2, RENDER_SIZE[1] - 2), (255, 255, 255), 3)
    render_ui(screen)
    if user_id is not None:
        render_ar(screen)

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

        id, confidence = recognizer.predict(face_img)

        if confidence >= 100: # Unknown face
            id = 0

        detected_user = users[id]
        detected_user.observe((image.shape[1], image.shape[0]), detected_face, face_img)
        updated_face_id = id

    return updated_face_id

if __name__ == "__main__":
    cv.startWindowThread()
    cv.namedWindow('image')
    cv.setWindowProperty('image', 0, 1)
    cv.moveWindow('image', 0, -SCREEN_SIZE[1])

    camera = CameraReader(True)

    camera.begin_reading()

    while True:
        main_loop(camera)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.end_reading()
    cv.destroyAllWindows()
