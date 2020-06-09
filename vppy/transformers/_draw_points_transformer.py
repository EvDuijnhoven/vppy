import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DrawPointsTransformer(BaseEstimator, TransformerMixin):
    """Draws points as circles on top of the image"""

    def __init__(self, points, weights=None, colour=(0, 255, 255), size=10):
        self.points = points
        self.weights = weights
        self.colour = colour
        self.size = size

    def fit(self, X: np.ndarray, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input not an image")
        if not len(X.shape) == 3:
            raise ValueError("Input image has the wrong shape")

        # If weights are supplied change size based on weight.
        if self.weights:
            # Scale from zero to one
            scales = self.weights / max(self.weights)
            self.sizes_ = np.ceil(self.size * scales + 1).astype(int)

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        for ind, point in enumerate(self.points):
            # Get individual or global size
            size = self.sizes_[ind] if hasattr(self, 'sizes_') else self.size

            cv2.circle(X, point, 0, self.colour, size)
        return X
