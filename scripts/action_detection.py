import numpy as np
import time
import collections

class ActionDetector:
    def __init__(self, callback):
        self._clear_histories()
        self.callback = callback

    def _clear_histories(self):
        self.last_time = None
        self.px_hist = collections.deque([0], 3)
        self.py_hist = collections.deque([0], 3)
        self.position = (0, 0)
        self.vx_hist = collections.deque([0], 20)
        self.vy_hist = collections.deque([0], 20)

    def update(self, position):
        if self.last_time is None:
            self.last_time = time.time()
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time

        lpx, lpy = self.position
        
        self.px_hist.append(position[0]); self.py_hist.append(position[1])
        px = np.mean(self.px_hist); py = np.mean(self.py_hist)
        self.position = (px, py)

        vx = (px - lpx) * elapsed * 50
        vy = (py - lpy) * elapsed * 100
        self.vx_hist.append(vx); self.vy_hist.append(vy)

        stdv_vx = np.std(self.vx_hist) * 100
        mean_vx = np.mean(self.vx_hist) * 100
        stdv_vy = np.std(self.vy_hist) * 100
        mean_vy = np.mean(self.vy_hist) * 100

        if stdv_x > 30 and abs(mean_vx) < 100:
            self.callback("shake")
            self._clear_histories()
        elif stdv_y > 25 and abs(mean_vy) < 100:
            self.callback("nod")
            self._clear_histories()

    def get_position(self):
        return self.position

    def get_velocity(self):
        return np.mean(self.vx_hist), np.mean(self.vy_hist)

    def get_velocity_stdev(self):
        return np.std(self.vx_hist) * 100, np.std(self.vy_hist) * 100
