import numpy as np
import os

from typing import List, Tuple

from smg.pyxnect import XNect
from smg.skeletons import Skeleton


class SkeletonDetector:
    """A 3D skeleton detector based on XNect."""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, exe_dir: str = "D:/xnect/bin/Release"):
        # Change to the XNect executable directory. The XNect guys hard-coded the paths for some reason (argh!).
        os.chdir(exe_dir)

        # Initialise XNect.
        self.__xnect = XNect()

        # Specify the keypoint names (see xnect_implementation.h).
        self.__keypoint_names = {
            0: "Head TOP",
            1: "Neck",
            2: "RShoulder",
            3: "RElbow",
            4: "RWrist",
            5: "LShoulder",
            6: "LElbow",
            7: "LWrist",
            8: "RHip",
            9: "RKnee",
            10: "RAnkle",
            11: "LHip",
            12: "LKnee",
            13: "LAnkle",
            14: "Root",
            15: "Spine",
            16: "Head",
            17: "RHand",
            18: "LHand",
            19: "RFoot",
            20: "LFoot"
        }

        # TODO

        # Enable debugging if requested.
        self.__debug = debug

    # PUBLIC METHODS

    def detect_skeletons(self, image: np.ndarray) -> Tuple[List[Skeleton], np.ndarray]:
        pass
