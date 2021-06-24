import cv2
import numpy as np
import os
import vg

from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple

from smg.pyxnect import XNect
from smg.skeletons import Keypoint, Skeleton3D
from smg.utility import GeometryUtil


class SkeletonDetector:
    """A 3D skeleton detector based on XNect."""

    # CONSTRUCTOR

    def __init__(self, *, exe_dir: str = "D:/xnect/bin/Release"):
        """
        Construct a 3D skeleton detector based on XNect.

        .. note::
            The reason for requiring the directory containing the XNect executable is that the
            XNect implementers hard-coded the paths (for some reason), so we need to change to
            this directory before trying to initialise XNect.

        :param exe_dir:     The directory containing the XNect executable.
        """
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
            16: "Nose",    # XNect calls this "Head", but we use "Nose" for consistency with other detectors
            17: "RHand",
            18: "LHand",
            19: "RFoot",
            20: "LFoot"
        }  # type: Dict[int, str]

        # Specify which keypoints are joined to form bones.
        self.__keypoint_pairs = [
            (self.__keypoint_names[i], self.__keypoint_names[j]) for i, j in [
                (0, 16), (1, 2), (1, 5), (1, 14), (1, 16), (2, 3), (3, 4), (4, 17), (5, 6), (6, 7), (7, 18),
                (8, 9), (8, 14), (9, 10), (11, 12), (11, 14), (12, 13)
            ]
        ]  # type: List[Tuple[str, str]]

        # Change to the XNect executable directory.
        os.chdir(exe_dir)

        # Initialise XNect.
        self.__xnect = XNect()

    # PUBLIC METHODS

    def detect_skeletons(self, image: np.ndarray, world_from_camera: np.ndarray, *, use_xnect_poses: bool = False,
                         visualise: bool = False) -> Tuple[List[Skeleton3D], np.ndarray]:
        """
        Detect 3D skeletons in an RGB image using XNect.

        :param image:               The RGB image.
        :param world_from_camera:   The camera pose.
        :param use_xnect_poses:     Whether to use the joint poses produced by XNect.
        :param visualise:           Whether to make the output visualisation.
        :return:                    A tuple consisting of the detected 3D skeletons and the output visualisation
                                    (if requested).
        """
        # Use XNect to detect any people in the image.
        # TODO: Figure out whether this copy is actually necessary.
        self.__xnect.process_image(image.copy())

        # Make the actual skeletons, and also the output visualisation if requested.
        skeletons = []                # type: List[Skeleton3D]
        visualisation = image.copy()  # type: np.ndarray

        # For each person index:
        for person_id in range(self.__xnect.get_num_of_people()):
            # If a person was detected with this index:
            if self.__xnect.is_person_active(person_id):
                # Construct the keypoints for the person's skeleton.
                skeleton_keypoints = {}  # type: Dict[str, Keypoint]

                # TODO
                midhip_w_t_c = np.eye(4)  # type: np.ndarray
                midhip_w_t_c[0:3, 0:3] = SkeletonDetector.__from_xnect_global_rotation(
                    self.__xnect.get_skeleton_global_rotation(person_id)
                )
                midhip_w_t_c[0:3, 3] = SkeletonDetector.__from_xnect_position(
                    self.__xnect.get_joint3d_ik(person_id, 14), world_from_camera
                )
                global_keypoint_poses = {"MidHip": midhip_w_t_c}  # type: Dict[str, np.ndarray]

                # TODO
                local_keypoint_rotations = {}  # type: Dict[str, np.ndarray]

                # For each joint (ignoring the feet, as in the sample code, as they can be unstable):
                for joint_id in range(self.__xnect.get_num_of_3d_joints() - 2):
                    # Make a keypoint for the joint and add it to the dictionary.
                    name = self.__keypoint_names[joint_id]
                    position = SkeletonDetector.__from_xnect_position(
                        self.__xnect.get_joint3d_ik(person_id, joint_id), world_from_camera
                    )
                    skeleton_keypoints[name] = Keypoint(name, position)

                    # TODO
                    local_keypoint_rotations[name] = SkeletonDetector.__from_xnect_local_rotation(
                        self.__xnect.get_joint_local_rotation(person_id, joint_id)
                    )

                # Add a skeleton based on the keypoints to the list.
                if use_xnect_poses:
                    skeletons.append(Skeleton3D(
                        skeleton_keypoints, self.__keypoint_pairs, global_keypoint_poses, local_keypoint_rotations
                    ))
                else:
                    skeletons.append(Skeleton3D(skeleton_keypoints, self.__keypoint_pairs))

                # Update the output visualisation if requested.
                if visualise:
                    self.__draw_bones(visualisation, person_id)
                    self.__draw_joints(visualisation, person_id)

        return skeletons, visualisation

    # PRIVATE METHODS

    def __draw_bones(self, image: np.ndarray, person_id: int) -> None:
        """
        Draw the bones for the specified person onto an image the shape of the input image.

        :param image:       The image onto which to draw the bones (must have the same shape as the input image).
        :param person_id:   The index of the person whose bones are to be drawn.
        """
        # For each joint (ignoring the feet, as in the sample code, as they can be unstable):
        for joint_id in range(self.__xnect.get_num_of_3d_joints() - 2):
            # Try to look up the joint's parent. If it doesn't have one, continue.
            parent_id = self.__xnect.get_joint3d_parent(joint_id)  # type: int
            if parent_id == -1:
                continue

            # Compute the 2D positions of the bone's two endpoints.
            endpoint_a = self.__get_joint_pos_2d(person_id, joint_id)   # type: Tuple[int, int]
            endpoint_b = self.__get_joint_pos_2d(person_id, parent_id)  # type: Tuple[int, int]

            # Draw the bone.
            colour = self.__get_person_colour(person_id)  # type: List[int]
            cv2.line(image, endpoint_a, endpoint_b, colour, 4)

    def __draw_joints(self, image: np.ndarray, person_id: int) -> None:
        """
        Draw the joints for the specified person onto an image the shape of the input image.

        :param image:       The image onto which to draw the joints (must have the same shape as the input image).
        :param person_id:   The index of the person whose joints are to be drawn.
        """
        # Specify the parameters to pass to cv2.circle when drawing the joints.
        radius = 6      # type: int
        thickness = -1  # type: int

        # For each joint (ignoring the feet, as in the sample code, as they can be unstable):
        for joint_id in range(self.__xnect.get_num_of_3d_joints() - 2):
            # Compute the 2D position of the joint.
            pos = self.__get_joint_pos_2d(person_id, joint_id)  # type: Tuple[int, int]

            # Draw the joint.
            colour = self.__get_person_colour(person_id)  # type: List[int]
            cv2.circle(image, pos, radius, colour, thickness)

    def __get_joint_pos_2d(self, person_id: int, joint_id: int) -> Tuple[int, int]:
        """
        Get the position of the specified joint for the specified detected person.

        :param person_id:   The index of the person.
        :param joint_id:    The index of the joint.
        :return:            The position of the specified joint for the specified detected person.
        """
        pos3d = self.__xnect.get_joint3d_ik(person_id, joint_id)  # type: np.ndarray
        pos2d = self.__xnect.project_with_intrinsics(pos3d)       # type: np.ndarray
        return tuple(np.round(pos2d).astype(int))

    def __get_person_colour(self, person_id: int) -> List[int]:
        """
        Get the colour assigned to the specified person.

        .. note::
            This gets the person's colour from XNect, and converts it into a representation that can be used by OpenCV.

        :param person_id:   The person whose assigned colour we want to get.
        :return:            The colour assigned to the person, as a list of integers.
        """
        return [int(_) for _ in self.__xnect.get_person_colour(person_id)]

    # PRIVATE STATIC METHODS

    @staticmethod
    def __from_xnect_global_rotation(rot: np.ndarray) -> np.ndarray:
        """
        TODO

        :param rot: TODO
        :return:    TODO
        """
        return np.linalg.inv(rot) @ np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

    @staticmethod
    def __from_xnect_local_rotation(rot: np.ndarray) -> np.ndarray:
        """
        TODO

        :param rot: TODO
        :return:    TODO
        """
        result = Rotation.from_matrix(rot).as_rotvec()  # type: np.ndarray
        result[0] *= -1
        result[2] *= -1
        return Rotation.from_rotvec(result).as_matrix()

    @staticmethod
    def __from_xnect_position(pos: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
        """
        TODO

        :param pos:                 TODO
        :param world_from_camera:   TODO
        :return:                    TODO
        """
        result = pos / 1000  # type: np.ndarray
        result[0] *= -1
        result[1] *= -1
        return GeometryUtil.apply_rigid_transform(world_from_camera, result)
