from threading import Thread
import time
import numpy as np
import cv2 as cv

class CameraReader:
    def __init__(self):
        self.camera = cv.VideoCapture(0)
        
        self.width = self.camera.get(3)
        self.height = self.camera.get(4)

        self.frame = None

        self.thread = Thread(target=self._read_camera)
        self.thread.daemon = True

    def begin_reading(self):
        self.reading = True
        self.thread.start()

    def end_reading(self):
        self.reading = False
        time.sleep(1) # Make sure the thread ends
        camera.release()

    def get_img_dimensions(self):
        return (self.width, self.height)

    def _read_camera(self):
        while self.reading:
            ret, frame = self.camera.read()
            frame = cv.flip(frame, +1)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.frame = gray

    def get_frame(self):
        frame = self.frame
        self.frame = None
        return frame
