import numpy as np
import cv2 as cv

import time

PIXELS_AT_STANDARD_DIST = 100.
eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')

class Position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Face:
    def __init__(self):
        self.bounding_rect = None
        self.position = Position(0, 0, 0)
        self.kposition = KalmanPosition()

    def observe(self, face, face_img):
        self.bounding_rect = face

        (fx, fy, fw, fh) = face
        self.position.x = fx
        self.position.y = fy

        top_offset = int(fh * 0.2)
        height_offset = int(fh * 0.5)

        cropped_frame = face_img[top_offset:height_offset, :]

        eyes = eye_cascade.detectMultiScale(cropped_frame)

        if len(eyes) != 2:
            # print("faulty eye observation")
            self.kposition.observe(time.time(), self.position)
            return

        left_eye  = eyes[0] if eyes[0][0] > eyes[1][0] else eyes[1]
        right_eye = eyes[0] if eyes[0][0] < eyes[1][0] else eyes[1]

        (lx, ly, lw, lh) = left_eye
        (rx, ry, rw, rh) = right_eye

        z = 1 - (((lx + lw / 2) - (rx + rw / 2)) / PIXELS_AT_STANDARD_DIST)
        self.position.z = z

        self.kposition.observe(time.time(), self.position)

    def get_distance_color(self):
        face_dist = self.position.z

        closeness = min(-min(0, face_dist * 2), 1)
        farness = min(max(0, face_dist * 2), 1)
        perfectness = 1 - min(np.square(face_dist) * 2, 1)

        return (int(farness * 255),int(perfectness * 255),int(closeness * 255))


def KF_predict_step(mu, Sigma, A, R):
    predicted_mu = np.dot(A, mu) #9x1
    predicted_sigma = np.dot(np.dot(A, Sigma), A.T) + R #9x9
    return predicted_mu, predicted_sigma

def KF_measurement_update_step(pred_mu, pred_Sigma, z, C, Q):
    K_l = np.dot(pred_Sigma, C.T)
    K_r = np.linalg.inv(np.dot(np.dot(C, pred_Sigma), C.T) + Q)
    K = np.dot(K_l, K_r)
    corrected_mu = pred_mu + np.dot(K, (z - np.dot(C, pred_mu)))
    corrected_sigma = np.dot((np.identity(9) - np.dot(K, C)), pred_Sigma)
    return corrected_mu, corrected_sigma

class KalmanPosition():
    """Kalman Filter node"""

    def __init__(self):
        # Filter variables
        self.mu = None          # state mean
        self.Sigma = None       # state covariance
        self.R = None           # covariance for the process model
        self.Q = None           # covariance for the measurement model
        self.A = None           # matrix that predicts the new state based on the prior state
        self.C = None           # matrix that transforms states into observations

        self.last_time = None

        # Initialize constant filter values
        self.initialize_values()

    def observe(self, ts, position):
            # initialize filter if state is None and save the current stamp as last_time
            if self.mu is None:
                self.initialize_mu_and_sigma(position)
                self.last_time = ts
                return

            # do nothing until we have initialized the filter
            if self.last_time is None:
                return

            # compute elapsed time from last prediction
            delta_t = ts - self.last_time
            assert delta_t >= 0, "Negative delta_t = {}?".format(delta_t) # sanity check!

            # assemble A matrix: helps generate new state from prior state and elapsed time
            self.assemble_A_matrix(delta_t)

            # prediction step: predict new mean and covariance
            self.mu, self.Sigma = KF_predict_step(self.mu, self.Sigma, self.A, self.R)

            # save the time of when we made the prediction
            self.last_time = ts

            # assemble observation vector
            z = [[position.x], [position.y], [position.z]]

            # measurement update step: correct mean and covariance
            self.mu, self.Sigma = KF_measurement_update_step(self.mu, self.Sigma, z, self.C, self.Q)

    def value(self):
        return (int(self.mu[0][0]), int(self.mu[1][0]), int(self.mu[2][0]))

    def assemble_A_matrix(self, delta_t):
        """
        Method that assembles the A matrix for the KF_predict_step
        :param delta_t: elapsed time (in seconds) since last prediction
        """
        d = delta_t
        s = .5 * (delta_t ** 2)
        self.A = np.array([[1, 0, 0, d, 0, 0, s, 0, 0],
                           [0, 1, 0, 0, d, 0, 0, s, 0],
                           [0, 0, 1, 0, 0, d, 0, 0, s],
                           [0, 0, 0, 1, 0, 0, d, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, d, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, d],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def initialize_values(self):
        """
        Method that assembles the C matrix for the KF_measurement_step
        """
        self.C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0]])

        l = 300
        m = 40
        s = 30
        self.R = np.array([[l, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, l, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, l, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, m, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, m, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, m, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, s, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, s, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, s]])

        self.Q = np.array([[1000, 0, 0],
                           [0, 1000, 0],
                           [0, 0, 1000]])

    def initialize_mu_and_sigma(self, position):
        """
        Method that initializes the state (sets self.mu and self.Sigma).
        :param obs_msg Observation message with the latest measured position for the target
        """
        l = 1500
        m = 500
        s = 100
        self.mu = np.array([[position.x], [position.y], [position.z], [0], [0], [0], [0], [0], [0]])
        self.Sigma = np.array([[l, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, l, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, l, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, m, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, m, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, m, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, s, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, s, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, s]])
