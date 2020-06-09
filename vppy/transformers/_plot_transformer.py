import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PlotTransformer(BaseEstimator, TransformerMixin):
    """Writes the image to the output folder"""

    def __init__(self, image_name, suffix='', folder='images/output'):
        self.image_name = image_name
        self.suffix = suffix
        self.folder = folder

    def fit(self, X: np.ndarray, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input not an image")
        return self

    # Method that describes what we need this transformers to do
    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        cv2.imwrite(f'{self.folder}/{self.image_name}_{self.suffix}.jpg', X)
        return X
