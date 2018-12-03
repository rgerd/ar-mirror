import numpy as np
import time
import collections

action_cooldown_time = 1 # second
action_time_threshold = 0.5 # second

class ActionDetector:
    def __init__(self, callback):
        self.callback = callback
        self.px_hist = collections.deque([0], 2)
        self.py_hist = collections.deque([0], 2)
        self.last_seen_left = 0
        self.last_seen_right = 0
        self.last_seen_up = 0
        self.last_seen_down = 0
        self.last_action_time = 0

    def update(self, position):
        current_time = time.time()

        self.px_hist.append(position[0]); self.py_hist.append(position[1])

        px = np.mean(self.px_hist) * 100
        py = (np.mean(self.py_hist) - 0.2) * 100
        self.position = (px, py)

        if px >= 6:
            self.last_seen_right = current_time
        elif px <= -6:
            self.last_seen_left = current_time

        if py <= -1:
            self.last_seen_up = current_time
        elif py >= 8:
            self.last_seen_down = current_time        

        self.last_seen_up    = self.last_seen_up    if current_time - self.last_seen_up    < action_time_threshold else 0
        self.last_seen_down  = self.last_seen_down  if current_time - self.last_seen_down  < action_time_threshold else 0
        self.last_seen_left  = self.last_seen_left  if current_time - self.last_seen_left  < action_time_threshold else 0
        self.last_seen_right = self.last_seen_right if current_time - self.last_seen_right < action_time_threshold else 0

        action = None

        if self.last_seen_left != 0 and self.last_seen_right != 0:
            action = "shake"
            
        elif self.last_seen_up != 0 and self.last_seen_down != 0:
            action = "nod"
            
        if action is not None and current_time - self.last_action_time > action_cooldown_time:
            self.last_action_time = current_time
            self.callback(action)

    # For debugging
    def get_position(self):
        return self.position
