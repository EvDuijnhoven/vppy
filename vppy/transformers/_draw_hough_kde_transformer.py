import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DrawHoughKDETransformer(BaseEstimator, TransformerMixin):
    """Draws the KDE on an image"""

    def __init__(self, kde):
        self.kde = kde

    def fit(self, X: np.ndarray, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input not an image")

        # Transforms the kde to an gray image
        kde_image = (255 * self.kde / np.max(self.kde)).astype('uint8')

        # Convert to colour image
        kde_image = cv2.cvtColor(kde_image, cv2.COLOR_GRAY2RGB)

        # Remove the red and blue values to make the line green
        kde_image[:, :, [0, 2]] = np.zeros([kde_image.shape[0], kde_image.shape[1], 2])

        # Resize the kde_image to the input image and store it
        self.kde_image_ = cv2.resize(kde_image, X.shape[0:2])
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        # Plot the kde_image on top of the input image
        return cv2.addWeighted(X, 1, self.kde_image_, 1, 0)
