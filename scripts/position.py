import numpy as np

class Position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def value(self):
        return (self.x, self.y, self.z)

    def dist(self, other):
        return Position(other.x - self.x, other.y - self.y, other.z - self.z)

    def update(self, target, amt):
        _amt = 1 - amt
        return Position(
            target.x * amt + self.x * _amt,
            target.y * amt + self.y * _amt,
            target.z * amt + self.z * _amt
        )


    def __str__(self):
        return "[ {0.x}, {0.y}, {0.z} ]".format(self)

#########################

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

    def _predict_step(self, mu, Sigma, A, R):
        predicted_mu = A @ mu #9x1
        predicted_sigma = (A @ Sigma) @ A.T + R #9x9
        return predicted_mu, predicted_sigma

    def _measurement_update_step(self, pred_mu, pred_Sigma, z, C, Q):
        K_l = pred_Sigma @ C.T
        K_r = np.linalg.inv(((C @ pred_Sigma) @ C.T) + Q)
        K = K_l @ K_r
        corrected_mu = pred_mu + K @ (z - (C @ pred_mu))
        corrected_sigma = (np.identity(K.shape[0]) - (K @ C)) @ pred_Sigma
        return corrected_mu, corrected_sigma

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
            self.mu, self.Sigma = self._predict_step(self.mu, self.Sigma, self.A, self.R)

            # save the time of when we made the prediction
            self.last_time = ts

            # assemble observation vector
            z = [[position.x], [position.y], [position.z]]

            # measurement update step: correct mean and covariance
            self.mu, self.Sigma = self._measurement_update_step(self.mu, self.Sigma, z, self.C, self.Q)

    def value(self):
        if self.mu is None:
            return (0, 0, 0)
        return (self.mu[0][0], self.mu[1][0], self.mu[2][0])

    def assemble_A_matrix(self, delta_t):
        """
        Method that assembles the A matrix for the KF_predict_step
        :param delta_t: elapsed time (in seconds) since last prediction
        """
        d = delta_t
        self.A = np.array([[1, 0, 0, d, 0, 0],
                           [0, 1, 0, 0, d, 0],
                           [0, 0, 1, 0, 0, d],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

    def initialize_values(self):
        """
        Method that assembles the C matrix for the KF_measurement_step
        """
        self.C = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])

        l = 300
        m = 40
        self.R = np.array([[l, 0, 0, 0, 0, 0],
                           [0, l, 0, 0, 0, 0],
                           [0, 0, l, 0, 0, 0],
                           [0, 0, 0, m, 0, 0],
                           [0, 0, 0, 0, m, 0],
                           [0, 0, 0, 0, 0, m]])

        self.Q = np.array([[1000, 0, 0],
                           [0, 1000, 0],
                           [0, 0, 1000]])

    def initialize_mu_and_sigma(self, position):
        """
        Method that initializes the state (sets self.mu and self.Sigma).
        :param position Latest measured position for the target
        """
        l = 1500
        m = 500
        self.mu = np.array([[position.x], [position.y], [position.z], [0], [0], [0]])
        self.Sigma = np.array([[l, 0, 0, 0, 0, 0],
                               [0, l, 0, 0, 0, 0],
                               [0, 0, l, 0, 0, 0],
                               [0, 0, 0, m, 0, 0],
                               [0, 0, 0, 0, m, 0],
                               [0, 0, 0, 0, 0, m]])
