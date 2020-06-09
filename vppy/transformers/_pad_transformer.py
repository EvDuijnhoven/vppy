import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PadTransformer(BaseEstimator, TransformerMixin):
    """Adds padding to the image"""

    def __init__(self, padding):
        self.padding = padding

    def fit(self, X: np.ndarray, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input not an image")
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        # Add padding
        shape = list(X.shape)
        shape[0] = shape[0] + 2 * self.padding
        shape[1] = shape[1] + 2 * self.padding
        base_image = np.zeros(shape, dtype=np.uint8)
        base_image[self.padding:shape[0] - self.padding, self.padding:shape[1] - self.padding] = X
        return base_image
