import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ColourSegmentTransformer(BaseEstimator, TransformerMixin):
    """Segments colours based on colour cluster using k-means"""

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input not an image")
        if not len(X.shape) == 3:
            raise ValueError("Input image has the wrong shape")
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        Z = X.reshape((-1, 3))

        # Convert to np.float32
        Z = np.float32(Z)

        # Define criteria, number of clusters(n_clusters) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, self.n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        return res.reshape((X.shape))