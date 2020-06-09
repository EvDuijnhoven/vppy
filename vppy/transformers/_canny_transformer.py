import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CannyTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X: np.ndarray, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input not an image")
        if not len(X.shape) == 3:
            raise ValueError("Input image has the wrong shape")
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        # Transform image to gray scale
        gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

        # Add Gaussian blur to remove small anomaly edges
        blur = cv2.GaussianBlur(gray, (3, 3), 20)

        # Get edges using Canny
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        return edges
