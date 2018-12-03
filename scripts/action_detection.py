import numpy as np
import time
import collections

action_time_threshold = 1.5 # seconds
action_cooldown_time = 1.5 # seconds

class ActionDetector:
    def __init__(self, callback):
        self.callback = callback
        self.px_hist = collections.deque([0], 2)
        self.py_hist = collections.deque([0], 2)
        self.position = (0, 0)
        self.last_seen_left = 0
        self.last_seen_right = 0
        self.last_seen_up = 0
        self.last_seen_down = 0
        self.last_action_time = 0
        self.last_time = None

    def update(self, position):
        if self.last_time is None:
            self.last_time = time.time()

        current_time = time.time()
        # print(current_time - self.last_time)
        self.last_time = current_time

        self.px_hist.append(position[0]); self.py_hist.append(position[1])

        px = np.mean(self.px_hist) * 100
        py = (np.mean(self.py_hist) - 0.2) * 100
        self.position = (px, py)

        if px >= 3: # 7:
            self.last_seen_right = current_time
        if px <= -3: # -7:
            self.last_seen_left = current_time

        if py <= 0:
            self.last_seen_up = current_time
        if py >= 5: # 6
            self.last_seen_down = current_time

        self.last_seen_up    = self.last_seen_up    if current_time - self.last_seen_up    < action_time_threshold else 0
        self.last_seen_down  = self.last_seen_down  if current_time - self.last_seen_down  < action_time_threshold else 0
        self.last_seen_left  = self.last_seen_left  if current_time - self.last_seen_left  < action_time_threshold else 0
        self.last_seen_right = self.last_seen_right if current_time - self.last_seen_right < action_time_threshold else 0

        action = None

        if self.last_seen_up != 0 and self.last_seen_down != 0:
            self.last_seen_up = 0 
            self.last_seen_down = 0
            action = "nod"

        elif self.last_seen_left != 0 and self.last_seen_right != 0:
            self.last_seen_left = 0 
            self.last_seen_right = 0
            action = "shake"

        if action is not None and current_time - self.last_action_time > action_cooldown_time:
            self.last_action_time = current_time
            self.callback(action)

    # For debugging
    def get_position(self):
        return self.position
