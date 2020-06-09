import cv2
import numpy as np
from sklearn.base import BaseEstimator
from ._hough_line import HoughLine
from ._hough_transform import HoughTransform


class HoughLinesEstimator(BaseEstimator):
    """Retrieves Hough lines from a binary image"""
    def __init__(self, threshold=0.1, vertical_degrees_filter=10, weight_decay=0.95):
        self.threshold = threshold
        self.vertical_degrees_filter = vertical_degrees_filter
        self.weight_decay = weight_decay

    def fit(self, X: np.ndarray, y=None):
        """Calculates the Houghlines using cv2 method"""
        if not isinstance(X, np.ndarray):
            raise ValueError("Input not an image")
        if not len(X.shape) == 2:
            raise ValueError("Input image of wrong dimensions")

        lines = cv2.HoughLines(
            X,
            1,
            np.radians(3),
            int(max(X.shape) * self.threshold), # Threshold dependent om input size (percentage of the image)
            min_theta=np.radians(self.vertical_degrees_filter),
            max_theta=np.radians(180 - self.vertical_degrees_filter)
        )

        if lines is None:
            lines = []
        else:
            lines = np.squeeze(lines, axis=1)

        # Create hough lines with specific weights.
        # Note: The order in which hough lines are returned is based on their vote count.
        #       It would however be better to use the actual votecount.
        self.lines_ = [HoughLine(line[0], line[1], self.weight_decay ** indx) for indx, line in enumerate(lines)]
        self.hough_transform_ = HoughTransform(X.shape, self.lines_)

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> HoughTransform:
        self.fit(X)
        return self.hough_transform_
