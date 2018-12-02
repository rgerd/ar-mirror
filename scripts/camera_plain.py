import numpy as np
import cv2 as cv
from camera_reader import CameraReader

def main_loop(camera):
    frame = camera.get_frame()
    
    if frame is None: # Wait for next frame
        return

    cv.imshow('image', frame)

if __name__ == "__main__":
    cv.startWindowThread()
    cv.namedWindow('image')

    camera = CameraReader()

    camera.begin_reading()

    while True:
        main_loop(camera)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.end_reading()
    cv.destroyAllWindows()
