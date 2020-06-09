import numpy as np
from typing import Tuple


class HoughLine():
    """Line defined by it's Hesse normal form"""

    def __init__(self, rho: float, theta: float, weight: float):
        self.rho = rho
        self.theta = theta
        self.weight = weight

    def distance(self, point: Tuple[float, float]):
        """Finds the distance between a line in Hesse normal form and a euclidean point.

        Returns distance between a line and a point https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        x, y = point

        return np.abs(x * np.cos(self.theta) + y * np.sin(self.theta) - self.rho)

    def add_padding(self, padding):
        # Add padding to hough lines by
        # Note: x0, y0 on the line now computes for x0 + padding, y0 + padding
        #   and angle does not change.
        self.rho = self.rho + padding * np.cos(self.theta) + padding * np.sin(self.theta)