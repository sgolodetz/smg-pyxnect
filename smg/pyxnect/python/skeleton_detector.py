import cv2
import numpy as np
import os

from typing import List, Tuple

from smg.pyxnect import XNect
from smg.skeletons import Skeleton
from smg.utility import GeometryUtil


class SkeletonDetector:
    """A 3D skeleton detector based on XNect."""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, exe_dir: str = "D:/xnect/bin/Release"):
        # Specify the keypoint names (see xnect_implementation.h).
        self.__keypoint_names = {
            0: "Head Top",
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
            14: "MidHip",  # XNect calls this "Root", but we use "MidHip" for consistency with other detectors
            15: "Spine",
            16: "Nose",  # XNect calls this "Head", but we use "Nose" for consistency with other detectors
            17: "RHand",
            18: "LHand",
            19: "RFoot",
            20: "LFoot"
        }

        # Specify which keypoints are joined to form bones.
        self.__keypoint_pairs = [
            (self.__keypoint_names[i], self.__keypoint_names[j]) for i, j in [
                (1, 2), (1, 5), (1, 14), (1, 16), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (8, 14), (9, 10),
                (11, 12), (11, 14), (12, 13)
            ]
        ]

        # Change to the XNect executable directory. The XNect guys hard-coded the paths for some reason (argh!).
        os.chdir(exe_dir)

        # Initialise XNect.
        self.__xnect = XNect()

        # Enable debugging if requested.
        self.__debug = debug

    # PUBLIC METHODS

    def detect_skeletons(self, image: np.ndarray, world_from_camera: np.ndarray, *,
                         visualise: bool = False) -> Tuple[List[Skeleton], np.ndarray]:
        # Prepare the XNect input image.
        height, width = image.shape[:2]
        xnect_input_size = (2048, 2048)

        if width > height:
            offset = (width - height) // 2
            xnect_input_image = cv2.resize(image[:, offset:width-offset, :], xnect_input_size)
        elif height > width:
            offset = (height - width) // 2
            xnect_input_image = cv2.resize(image[offset:height-offset, :], xnect_input_size)
        else:
            xnect_input_image = image.copy()

        # Use XNect to detect any people in the image.
        self.__xnect.process_image(xnect_input_image)

        # Make the actual skeletons, and also the output visualisation if requested.
        skeletons = []
        visualisation = image.copy()

        for person_id in range(self.__xnect.get_num_of_people()):
            if self.__xnect.is_person_active(person_id):
                skeleton_keypoints = {}

                # Note: As in the sample code, we ignore the feet, as they can be unstable.
                for joint_id in range(self.__xnect.get_num_of_3d_joints() - 2):
                    name = self.__keypoint_names[joint_id]
                    position = self.__xnect.get_joint3d_ik(person_id, joint_id) / 1000
                    position[0] *= -1
                    position[1] *= -1
                    position = GeometryUtil.apply_rigid_transform(world_from_camera, position)
                    skeleton_keypoints[name] = Skeleton.Keypoint(name, position)

                skeletons.append(Skeleton(skeleton_keypoints, self.__keypoint_pairs))

                # Update the output visualisation if requested.
                if visualise:
                    self.__draw_bones(visualisation, person_id)
                    self.__draw_joints(visualisation, person_id)

        return skeletons, visualisation

    # PRIVATE METHODS

    def __draw_bones(self, image: np.ndarray, person_id: int) -> None:
        # Note: As in the sample code, we ignore the feet, as they can be unstable.
        for joint_id in range(self.__xnect.get_num_of_3d_joints() - 2):
            parent_id = self.__xnect.get_joint3d_parent(joint_id)
            if parent_id == -1:
                continue

            # lookup 2 connected body/hand parts
            part_a = SkeletonDetector.__scale_to_image(
                self.__xnect.project_with_intrinsics(self.__xnect.get_joint3d_ik(person_id, joint_id)), image
            )
            part_b = SkeletonDetector.__scale_to_image(
                self.__xnect.project_with_intrinsics(self.__xnect.get_joint3d_ik(person_id, parent_id)), image
            )

            if part_a[0] <= 0 or part_a[1] <= 0 or part_b[0] <= 0 or part_b[1] <= 0:
                continue

            colour = self.__get_person_colour(person_id)
            cv2.line(image, part_a, part_b, colour, 4)

    def __draw_joints(self, image: np.ndarray, person_id: int) -> None:
        radius = 6
        thickness = -1

        # Note: As in the sample code, we ignore the feet, as they can be unstable.
        for joint_id in range(self.__xnect.get_num_of_3d_joints() - 2):
            pos = SkeletonDetector.__scale_to_image(
                self.__xnect.project_with_intrinsics(self.__xnect.get_joint3d_ik(person_id, joint_id)), image
            )
            colour = self.__get_person_colour(person_id)
            cv2.circle(image, pos, radius, colour, thickness)

    def __get_person_colour(self, person_id) -> List[int]:
        return [int(_) for _ in self.__xnect.get_person_colour(person_id)]

    # PRIVATE STATIC METHODS

    @staticmethod
    def __scale_to_image(pos: np.ndarray, image: np.ndarray) -> Tuple[int, int]:
        height, width = image.shape[:2]
        # TODO: Correct for cropping.
        return tuple(np.round((pos[0] * width / 1024, pos[1] * height / 1024)).astype(int))
