import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SegmentTransformer(BaseEstimator, TransformerMixin):
    """Removes segment from the image"""

    def __init__(self, alpha=0.15, beta=0.6):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X: np.ndarray, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input not an image")
        return self

    def transform(self, X, y=None):
        height, width = X.shape

        # Create a triangular polygon based on alpha and beta at the bottom of the page.
        polygons = np.array([[
            (int(-self.alpha * width), height),
            (int(width * (1+self.alpha)), height),
            (int(width / 2), int(height * self.beta))
        ]])

        # Creates an image filled with 255 for outside of the polygon
        mask = np.ones_like(X) * 255
        cv2.fillPoly(mask, polygons, 0)

        # Only keep the values outside of the triangle
        return cv2.bitwise_and(X, mask)
