import numpy as np
from sklearn.base import BaseEstimator
from ._hough_line import HoughLine
from typing import List, Tuple


class HoughLinesKDE(BaseEstimator):
    """Computes a KDE (Kernel Density Estimate) for a list of HoughLines"""

    def __init__(self, shape, beta=5, kde_width=200, kde_height=200):
        self.shape = shape
        self.beta = beta
        self.kde_width = kde_width
        self.kde_height = kde_height

    def fit(self, X: List[HoughLine], y=None):
        # Define the bandwidth relative to the image size
        bandwidth = 100 * self.beta / max(self.shape)
        img_height, img_width = self.shape

        kde = np.zeros((self.kde_height, self.kde_width), dtype=np.float32)
        # For every HoughLine compute and add its kernel to the total kde
        for line in X:
            distance = self._get_distance_matrix(line, bandwidth, img_width, img_height)

            # Calculate the kernel over the distance matrix
            kernel = self._epanechnikov_matrix(distance)

            # Add the weighted value of the kernel to the total kde.
            kde = np.add(kde, kernel * line.weight)

        # Get the max index of the kde as the vanishing point
        y, x = np.unravel_index(kde.argmax(), kde.shape)
        self.vp_ = self._scale_point(x, y, img_width, img_height)
        self.kde_ = kde

        return self

    def _get_distance_matrix(self, line: HoughLine, bandwidth: float, img_width: int, img_height: int) -> np.ndarray:
        """Calculates the distance matrix for every point in the grid to the HoughLine"""

        # Calculate the rho for every point in the matrix, defined by x cos(theta) + y sin(theta)
        X = np.repeat([np.linspace(0, 1, self.kde_width) * img_width * np.cos(line.theta)], self.kde_height, axis=0)
        Y = np.transpose(
            np.repeat([np.linspace(0, 1, self.kde_height) * img_height * np.sin(line.theta)], self.kde_width, axis=0))

        # Center every point around rho, smooth using the bandwidth.
        return (np.add(X, Y) - line.rho) * bandwidth

    def _scale_point(self, x, y, img_width: int, img_height: int) -> Tuple[int, int]:
        """Scales points to the size of the image"""
        return int(np.round((x + 0.5) * img_width / self.kde_width)), int(
            np.round((y + 0.5) * img_height / self.kde_height))

    @staticmethod
    def _epanechnikov_matrix(distance):
        """Epanechnikov kernel; https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use"""
        return 0.75 * (1 - np.minimum(np.square(distance), 1))
