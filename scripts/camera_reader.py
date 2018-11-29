from threading import Thread
import time
import numpy as np
import cv2 as cv

class CameraReader:
    def __init__(self, convert_gray=False):
        self.camera = cv.VideoCapture(0)

        self.camera_width = int(self.camera.get(3))
        self.camera_height = int(self.camera.get(4))
        self.frame_width = int(self.camera_width * 0.3)
        self.frame_height = int(self.camera_height * 0.3)

        self.frame = None
        self.convert_gray = convert_gray

        self.thread = Thread(target=self._read_camera)
        self.fps = 0

    def begin_reading(self):
        self.reading = True
        self.thread.start()

    def end_reading(self):
        self.reading = False
        time.sleep(1) # Make sure the thread ends
        self.camera.release()

    def get_img_dimensions(self):
        return (self.frame_width, self.frame_height)

    def _read_camera(self):
        while self.reading:
            time_start = time.time()
            ret, frame = self.camera.read()
            frame = cv.flip(frame, +1)
            if self.convert_gray:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.frame = cv.resize(frame, (self.frame_width, self.frame_height))
            self.frame_time = time.time() - time_start

    def get_frame(self):
        frame = self.frame
        self.frame = None
        return frame

    def get_fps(self):
        return int(1 / self.frame_time)
