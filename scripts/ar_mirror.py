import numpy as np
import cv2 as cv
from camera_reader import CameraReader
from ui_render import render_ui

RENDER_SIZE = (540, 765)
SCREEN_SIZE = (1080, 1530)

def main_loop(camera):
    frame = camera.get_frame()
    
    if frame is None: # Wait for next frame
        return

    screen = np.zeros((RENDER_SIZE[1], RENDER_SIZE[0], 3), dtype=np.uint8)  
    render_ui(screen)
    
    cv.rectangle(screen, (0, 0), (RENDER_SIZE[0] - 2, RENDER_SIZE[1] - 2), (255, 255, 255), 3)

    screen = cv.resize(screen, SCREEN_SIZE, interpolation=0)

    cv.imshow('image', screen)

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
