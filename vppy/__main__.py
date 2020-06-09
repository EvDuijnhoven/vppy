import os
import cv2
import numpy as np
import argparse
from .transformers import ResizeTransformer, CannyTransformer, PlotTransformer, SegmentTransformer, PadTransformer,\
    DrawHoughKDETransformer, DrawPointsTransformer, ColourSegmentTransformer
from .hough import HoughTransform, HoughLinesEstimator
from sklearn.pipeline import Pipeline


def run(args):
    """Main method, parses all images in the given folder"""
    filepaths = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if
                 os.path.isfile(os.path.join(args.input_folder, f)) and f.endswith('.jpg')]
    for filepath in filepaths:
        # If it is not an image continue
        if not filepath.endswith(".jpg"):
            continue
        image_name = os.path.basename(filepath).split('.')[0]
        img = cv2.imread(filepath)
        parse_image(args, img, image_name)


def parse_image(args, root_img: np.ndarray, image_name: str):
    """Parse a single image, get the hough lines and find the vanishing points"""
    # Create hough pipeline that transforms the image and finally predicts the hough lines
    # Note: This pipeline is saving intermediate steps as image files in the output folder
    hough_pipeline = Pipeline(steps=[
        ('image_resizer', ResizeTransformer(args.shape)),
        # ('colour_masker', ColourSegmentTransformer(3)),
        # ('plot_colour_mask_image', PlotTransformer(image_name, suffix="colour", folder=args.output_folder)),
        ('canny_image', CannyTransformer()),
        ('segment_image', SegmentTransformer() if args.segment_canny else None),
        ('plot_canny_image', PlotTransformer(image_name, suffix="canny", folder=args.output_folder)),
        ('hough_transform', HoughLinesEstimator(
            threshold=args.hough_threshold,
            weight_decay=args.weight_decay,
            vertical_degrees_filter=args.degrees_filter
        )),
    ])
    # Get the pipeline results and filter them according to argument settings
    hough_transform = hough_pipeline.fit_predict(root_img)
    hough_transform.filter_horizontal_lines(degrees=args.degrees_filter)
    hough_transform.limit_lines(args.hough_limit)
    hough_transform.group_lines(r=args.hough_group_radius)
    if args.cluster_hough_lines:
        hough_transform.cluster_lines()

    # Add padding to the hough transform
    hough_transform.add_padding(args.padding)

    # Get the vanishing points with the chosen method
    vps, reference_transformer = METHODS[args.method](args, hough_transform)

    # Plot decimal coordinates of the vanish points
    print([((x - args.padding )/args.img_width, (y - args.padding)/args.img_height) for x,y in vps])

    # Plot and print on the original image
    Pipeline(steps=[
        ('image_resizer', ResizeTransformer(args.shape)),
        ('pad_image', PadTransformer(args.padding)),
        # ('plot_pad_image', PlotTransformer(image_name, suffix="padded_orig"))
        ('add_reference', reference_transformer),
        ('add_vanishing_points', DrawPointsTransformer(vps, colour=(255, 255, 0))),
        ('plot_final_image', PlotTransformer(image_name, suffix="final", folder=args.output_folder)),
    ]).fit_transform(root_img)

def parse_kde(args, hough_transform: HoughTransform):
    """Get vanishing points using the KDE method"""
    vps, kdes = hough_transform.find_vps_kdes(
        weight_threshold=args.weight_threshold,
        filter_threshold=args.filter_threshold,
        kde_width=args.kde_width,
        kde_height=args.kde_height,
        kde_beta=args.kde_beta
    )

    kde_transformer = DrawHoughKDETransformer(kdes[0].kde_) if kdes else None
    # Return the vppy and the transformer to draw the kde used
    return vps, kde_transformer


def parse_intersection(args, hough_transform: HoughTransform):
    """Get vanishing points using the intersection method"""
    vps = hough_transform.find_vps_intersections(
        weight_threshold=args.weight_threshold,
        filter_threshold=args.filter_threshold
    )
    intersections, weights = hough_transform.intersections()
    # Return the vppy and the transformer to draw the intersections used
    return vps, DrawPointsTransformer(intersections, weights=weights, size=10)


METHODS = {'kde': parse_kde, 'intersection': parse_intersection}


class Directory(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        path = value
        if not os.path.isdir(path):
            raise argparse.ArgumentError(self, f'{path} is not a valid or existing dir')
        if not os.access(path, os.R_OK):
            raise argparse.ArgumentError(self, f'{path} is not a readable dir')
        if not os.access(path, os.W_OK):
            raise argparse.ArgumentError(self, f'{path} is not a writable dir')
        setattr(namespace, self.dest, path)


class Range(argparse.Action):
    def __init__(self, min=None, max=None, *args, **kwargs):
        self.min = min
        self.max = max
        kwargs["metavar"] = f'({self.min}-{self.max})'
        super(Range, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        if not (self.min <= value <= self.max):
            raise argparse.ArgumentError(self, f'invalid choice: {value} (choose from ({self.min}-{self.max}))')
        setattr(namespace, self.dest, value)


def main():
    """Entry point method"""
    parser = argparse.ArgumentParser(
        description='Extract Vanishing points',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_folder', action=Directory, metavar="<path>",
                        help='Folder in which the input images are stored')
    parser.add_argument('output_folder', action=Directory, metavar="<path>",
                        help='Folder in which the intermediate and final images are stored')
    parser.add_argument('--img_width',  type=int, metavar="<int>", default=400,
                        help='Output image width')
    parser.add_argument('--img_height', type=int, metavar="<int>", default=400,
                        help='Output image height')
    parser.add_argument('--padding', type=int, default=100, min=0, max=1000, action=Range,
                        help='Padidng used around the image')
    parser.add_argument('--kde_width', type=int, metavar="<int>", default=400,
                        help='Kde grid width')
    parser.add_argument('--kde_height', type=int, metavar="<int>", default=400,
                        help='Kde grid height')
    parser.add_argument('--kde_beta', type=float, default=1, min=0, max=1, action=Range,
                        help='Kde beta manipulating the bandwidth of kde lines')
    parser.add_argument('--weight_decay', type=float, default=0.95, min=0.8, max=1, action=Range,
                        help='Weight decay used to weigh Hough Lines')
    parser.add_argument('--filter_threshold', type=float, default=0.3, min=0, max=1, action=Range,
                        help='Threshold for which to filter Hough lines near a vanishing point')
    parser.add_argument('--weight_threshold', type=float, default=0.1, min=0, max=1, action=Range,
                        help='Threshold for which weight relevance a new vanishing point should be searched')
    parser.add_argument('--hough_threshold', type=float, default=0.10, min=0, max=1, action=Range,
                        help='Threshold for the Hough Transform votes')
    parser.add_argument('--hough_limit', type=int, metavar="<int>", default=200,
                        help='Hough limit of input variables')
    parser.add_argument('--hough_group_radius', type=float, default=0.01, min=0, max=0.1, action=Range,
                        help='Radius used for Hough Line Groups')
    parser.add_argument('--cluster_hough_lines', default=False, action='store_true',
                        help='Cluster hough lines using Affinity Propagation')
    parser.add_argument('--segment_canny', default=False, action='store_true',
                        help='Remove bottom segment from the canny image')
    parser.add_argument('--degrees_filter', type=int, default=10,  min=0, max=45, action=Range,
                        help='Degrees of freedom in which horizontal and vertical lines are filtered')
    parser.add_argument('--method', type=str, default='kde', choices=METHODS.keys(),
                        help='Method used for predicting the vanishing points. (kde or intersection) ')
    args = parser.parse_args()
    args.shape = (args.img_width, args.img_height)
    run(args)


if __name__ == '__main__':
    main()