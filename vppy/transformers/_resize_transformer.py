import cv2
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ResizeTransformer(BaseEstimator, TransformerMixin):
    """Resize the image"""

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def fit(self, X: np.ndarray, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input not an image")
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return cv2.resize(X, self.input_shape)
