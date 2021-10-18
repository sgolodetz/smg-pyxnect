# noinspection PyPackageRequirements
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame

from argparse import ArgumentParser
from timeit import default_timer as timer
from typing import Callable, List, Optional, Tuple

from smg.comms.skeletons import SkeletonDetectionService
from smg.pyxnect import SkeletonDetector
from smg.skeletons import Skeleton3D


def make_frame_processor(skeleton_detector: SkeletonDetector, *, debug: bool = False, use_xnect_poses: bool) \
        -> Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[List[Skeleton3D], Optional[np.ndarray]]]:
    """
    Make a frame processor for a skeleton detection service that forwards to an XNect skeleton detector.

    :param skeleton_detector:   The XNect skeleton detector.
    :param debug:               Whether to print debug messages.
    :param use_xnect_poses:     Whether to use the joint poses produced by XNect.
    :return:                    The frame processor.
    """
    # noinspection PyUnusedLocal
    def detect_skeletons(colour_image: np.ndarray, depth_image: np.ndarray, world_from_camera: np.ndarray) \
            -> Tuple[List[Skeleton3D], Optional[np.ndarray]]:
        """
        Detect 3D skeletons in an RGB image using XNect.

        :param colour_image:        The RGB image.
        :param depth_image:         Passed in by the skeleton detection service, but ignored.
        :param world_from_camera:   The camera pose.
        :return:                    The detected 3D skeletons (the people mask will be rendered internally).
        """
        if debug:
            start = timer()

        # Detect the skeletons.
        skeletons, _ = skeleton_detector.detect_skeletons(
            colour_image, world_from_camera, use_xnect_poses=use_xnect_poses
        )

        # If we're using the computed keypoint orientation information rather than the local rotations provided by
        # XNect, remove it so that it will be recomputed by the client (this is necessary if we want to visualise
        # the keypoint orienters, since those can't be easily sent over the network).
        if not use_xnect_poses:
            skeletons = [s.make_bare() for s in skeletons]

        if debug:
            end = timer()

            # noinspection PyUnboundLocalVariable
            print("Detection Time: {}s".format(end - start))

        return skeletons, None

    return detect_skeletons


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, default=7852,
        help="the port on which the service should listen for a connection"
    )
    parser.add_argument(
        "--use_xnect_poses", action="store_true",
        help="whether to use the joint poses produced by XNect"
    )
    args = vars(parser.parse_args())  # type: dict

    # Initialise PyGame and create a hidden window so that we can use OpenGL.
    pygame.init()
    window_size = (1, 1)  # type: Tuple[int, int]
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.HIDDEN | pygame.OPENGL)

    # Construct the skeleton detector.
    skeleton_detector = SkeletonDetector()

    # Run the skeleton detection service.
    service = SkeletonDetectionService(
        make_frame_processor(skeleton_detector, use_xnect_poses=args["use_xnect_poses"]), args["port"],
        post_client_hook=skeleton_detector.restart
    )
    service.run()


if __name__ == "__main__":
    main()
