import itertools
import numpy as np
from typing import Tuple, List
from ._hough_lines_kde import HoughLinesKDE
from ._hough_line import HoughLine
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import KDTree
from scipy import stats


class HoughTransform():
    """Class containing the Hough Transform information and manipulations

    https://en.wikipedia.org/wiki/Hough_transform
    """

    def __init__(self, shape: Tuple[int, int], lines: List[HoughLine]):
        self.shape = shape
        self.lines = lines

    @property
    def __len__(self):
        return len(self.lines)

    @property
    def weight(self):
        return self.sum_line_weight(self.lines)

    @property
    def features(self):
        """Feature set used for clustering"""
        # Divide by image size to be "independent" on image size.
        return [[line.theta / np.pi, line.rho / max(self.shape)] for line in self.lines]

    def add_padding(self, padding):
        for line in self.lines:
            line.add_padding(padding)

        self.shape = tuple(dim + 2*padding for dim in self.shape)

    def filter_horizontal_lines(self, degrees: int = 5):
        """Filter all horizontal HoughLines from this HoughTransform"""
        min_theta = np.radians(90 - degrees)
        max_theta = np.radians(90 + degrees)
        self.lines = [line for line in self.lines if min_theta > line.theta or line.theta > max_theta]
        return self

    def limit_lines(self, limit: int = None):
        """Limit the total amount of lines in this HoughTransform"""
        if int:
            self.lines = self.lines[0:limit]
        return self

    def cluster_lines(self):
        """Cluster lines using AffinityPropagation"""

        if self.__len__ < 2 :
            return self

        cluster = AffinityPropagation(damping=0.7, random_state=0).fit(self.features)

        # Create cluster representatives, sum the weights
        lines = []
        for i in range(max(cluster.labels_)):
            cluster_lines = np.array(self.lines)[cluster.labels_ == i]
            weights = [line.weight for line in cluster_lines]

            # Pick the line with the highest weight as the cluster representative
            cluster_rep = cluster_lines[np.argmax(weights)]

            lines.append(HoughLine(cluster_rep.rho, cluster_rep.theta, sum(weights)))
        self.lines = lines
        return self

    def group_lines(self, r=0.3):
        """Group lines that that have features within a certain radius of each other"""
        # If no radius every line only is within it's own group
        if self.__len__ < 2 or r == 0:
            return self

        tree = KDTree(np.array(self.features), leaf_size=2)

        # Parse through all lines, combine lines close to the line.
        # Note: Sort to start with the highest weighted lines
        sorted_lines = sorted(self.lines, key=lambda x: x.weight)
        indexes = list(range(len(sorted_lines)))
        lines = []
        while len(indexes) > 0:
            ind = indexes[0]
            # Pick the line with the highest weight as the cluster representative
            line = sorted_lines[ind]
            feature = self.features[ind]

            # Get all indexes that are within the radius and not already grouped
            nn_ind = [i for i in tree.query_radius([feature], r=r)[0] if i in indexes]

            # Create a Hough line representa
            nn_lines = np.array(sorted_lines)[nn_ind]
            lines.append(HoughLine(line.rho, line.theta, self.sum_line_weight(nn_lines)))

            # Remove parsed lines from remaining indexes.
            for i in nn_ind:
                indexes.remove(i)

        self.lines = lines
        return self

    def intersections(self, lines: List[HoughLine] = None):
        """Get the intersections of all the hough lines"""
        if not lines:
            lines = self.lines
        intersections = []
        weights = []
        # Loop through all combinations of lines
        for ind, line1 in enumerate(lines):
            for line2 in lines[ind + 1:]:
                intersection, weight = self.get_line_intersection(line1, line2)
                if intersection:
                    intersections.append(intersection)
                    weights.append(weight)
        return intersections, weights

    @staticmethod
    def sum_line_weight(lines: List[HoughLine]):
        """Calculate the sum of a list of HoughLines"""
        return sum([line.weight for line in lines])

    @staticmethod
    def get_line_intersection(line1: HoughLine, line2: HoughLine):
        """Calculate the intersection point of two HoughLines"""
        # Return None if lines are parallel
        if line1.theta == line2.theta:
            return None, None

        # Using Cramer's rule via matrix inversion (https://en.wikipedia.org/wiki/Intersection_(Euclidean_geometry))
        A = [
            [np.cos(line1.theta), np.sin(line1.theta)],
            [np.cos(line2.theta), np.sin(line2.theta)]
        ]
        b = [line1.rho, line2.rho]

        # Calculate the weight of the intersection based on the weight of the lines and their relative angle
        weight = np.sqrt(line1.weight*line2.weight*np.abs(np.sin(line1.theta - line2.theta)))

        return tuple(np.linalg.inv(A).dot(b).astype(int)), weight

    def filter_lines_on_vp(self, lines, vp: Tuple[int, int], threshold: float = 0.3):
        """Filter lines that are within a threshold distance of the vanishing point"""
        # Divide by image size to be "independent" on image size.
        return [line for line in lines if line.distance(vp) / max(self.shape) > threshold]

    def find_vps_intersections(self, weight_threshold=0.1, filter_threshold=0.1):
        """Finds the vanishing points based on the HoughTransform Line intersections"""
        vps = []
        lines = self.lines

        # Loop while there are enough lines and their weight is still relevant.
        while len(lines) > 2 \
                and self.sum_line_weight(lines) > weight_threshold * self.weight:
            # Get all intersections of the remaining lines
            intersections, weights = self.intersections(lines)

            # Calculate the KDE for every intersection
            kernel = stats.gaussian_kde(np.transpose(np.array(intersections)), weights=np.array(weights))
            intersection_densities = kernel.evaluate(np.transpose(np.array(intersections)))

            # Retrieve the intersection point with the highest density
            vp = intersections[intersection_densities.argmax()]

            # Filter the lines that where contributing to the vanishing point
            lines = self.filter_lines_on_vp(lines, vp, threshold=filter_threshold)
            vps.append(vp)
        return vps

    def find_vps_kdes(self, weight_threshold=0.1, filter_threshold=0.1, kde_width=200, kde_height=200, kde_beta=1):
        """Finds the vanishing points based on the HoughTransform Line kernel density estimation"""
        kdes = []
        lines = self.lines

        # Loop while there are enough lines and their weight is still relevant.
        while len(lines) > 2 \
                and self.sum_line_weight(lines) > weight_threshold * self.weight:
            # Calculate the Hough Line KDE for the remaining lines
            h_kde = HoughLinesKDE(
                self.shape,
                beta=kde_beta,
                kde_width=kde_width,
                kde_height=kde_height
            ).fit(lines)

            # Filter the lines that where contributing to the vanishing point
            lines = self.filter_lines_on_vp(lines, h_kde.vp_, threshold=filter_threshold)
            kdes.append(h_kde)
        return [kde.vp_ for kde in kdes], kdes
