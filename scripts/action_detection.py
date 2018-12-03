import numpy as np
import time
import collections

action_cooldown_time = 1 # second

class ActionDetector:
    def __init__(self, callback):
        self._clear_histories()
        self.callback = callback
        self.px_hist = collections.deque([0], 3)
        self.py_hist = collections.deque([0], 3)
        self.position = (0, 0)
        self.last_time = None
        self.last_action_time = 0

    def _clear_histories(self):
        self.vx_hist = collections.deque([0], 10)
        self.vy_hist = collections.deque([0], 10)

    def update(self, position):
        current_time = time.time()

        self.px_hist.append(position[0]); self.py_hist.append(position[1])

        if self.last_time is None:
            self.last_time = current_time
            return

        elapsed = current_time - self.last_time
        self.last_time = current_time

        lpx, lpy = self.position # last px, py
        px = np.mean(self.px_hist)
        py = np.mean(self.py_hist)
        self.position = (px, py)

        vx = (px - lpx) * elapsed * 50
        vy = (py - lpy) * elapsed * 100
        self.vx_hist.append(vx); self.vy_hist.append(vy)

        stdv_vx = np.std(self.vx_hist) * 100
        mean_vx = np.mean(self.vx_hist) * 100
        stdv_vy = np.std(self.vy_hist) * 100
        mean_vy = np.mean(self.vy_hist) * 100

        action = None

        if stdv_vx > 50 and abs(mean_vx) < 100:
            action = "shake"
            
        elif stdv_vy > 35 and abs(mean_vy) < 100:
            action = "nod"
            
        if action is not None and current_time - self.last_action_time > action_cooldown_time:
            self.last_action_time = current_time
            self._clear_histories()
            self.callback(action)


    def get_position(self):
        return self.position

    def get_velocity(self):
        return np.mean(self.vx_hist), np.mean(self.vy_hist)

    def get_velocity_stdev(self):
        return np.std(self.vx_hist) * 100, np.std(self.vy_hist) * 100
