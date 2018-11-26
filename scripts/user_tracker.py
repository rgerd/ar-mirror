## DETERMINED REASONABLE EYE DISTANCE @ 1D: 100px
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

# # axis centered at the center of the mirror.
# # positive x to the right. y is up. z is out towards user.
# def pixel_to_center_frame(point):
#     point[0] -= window_width / 2
#     point[1] -= window_height / 2
#     return point

# def center_frame_to_pixel(point):
#     point[0] += window_width / 2
#     point[1] += window_height / 2
#     return point

# def mirror_point(perspective, location):
#     if perspective[2] <= 0:
#         print('invalid perspective: face is behind the mirror')
#         return (0,0)
#     if location[2] > 0:
#         print('invalid location: positive z position for location... flipping sign')
#         location[2] = -location[2]

#     # calculate for z = 0
#     (px, py, pz) = perspective
#     (lx, ly, lz) = location
#     x = px - ((px - lx) / float(pz - lz)) * pz
#     y = py - ((py - ly) / float(pz - lz)) * pz

#     return center_frame_to_pixel([x, y])

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
        detected_user.observe(detected_face, face_img)
        updated_face_id = id

    return updated_face_id

def render_ui(user, user_name, screen):
    # render face display
    color = user.get_distance_color()
    filtered_color = user.get_distance_color(filter=True)

    # calculate perspective (assuming camera is center of mirror)
    (x, y, z) = user.filtered_position.value()
    fw, fh = user.measured_size

    # normalize
    x += fw / 2.
    y += fw / 3.
    cv.circle(screen, (int(x), int(y)), 5, filtered_color, 2)

    # point = [0, 0, -2] # center of mirror with depth of 10
    # perspective = pixel_to_center_frame([x,y,z])
    # static_x, static_y = mirror_point(perspective, point)
    # cv.circle(screen, (int(static_x), int(static_y)), 5, (255, 255, 255), 2)

    cv.putText(screen, user_name, (int(x)+5,int(y)-20), font, 1, (255,255,255), 2)


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
    mirror_mode = '--mirror' in sys.argv

    camera = CameraReader(mirror_mode)

    camera.begin_reading()
    while True:
        main_loop(camera, mirror_mode)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.end_reading()
    cv.destroyAllWindows()
