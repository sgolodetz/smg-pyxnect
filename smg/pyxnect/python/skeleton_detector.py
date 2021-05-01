import cv2
import numpy as np
import os

from typing import Dict, List, Tuple

from smg.pyxnect import XNect
from smg.skeletons import Skeleton
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
        }

        # Specify which keypoints are joined to form bones.
        # FIXME: I haven't connected all of the joints yet.
        self.__keypoint_pairs = [
            (self.__keypoint_names[i], self.__keypoint_names[j]) for i, j in [
                (1, 2), (1, 5), (1, 14), (1, 16), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (8, 14), (9, 10),
                (11, 12), (11, 14), (12, 13)
            ]
        ]

        # Change to the XNect executable directory.
        os.chdir(exe_dir)

        # Initialise XNect.
        self.__xnect = XNect()

    # PUBLIC METHODS

    def detect_skeletons(self, image: np.ndarray, world_from_camera: np.ndarray, *,
                         visualise: bool = False) -> Tuple[List[Skeleton], np.ndarray]:
        """
        Detect 3D skeletons in an RGB image using XNect.

        :param image:               The RGB image.
        :param world_from_camera:   The camera pose.
        :param visualise:           Whether to make the output visualisation.
        :return:                    A tuple consisting of the detected 3D skeletons and the output visualisation
                                    (if requested).
        """
        # Prepare the XNect input image. This involves cropping it to be square, then upsampling it.
        height, width = image.shape[:2]
        offset = abs(width - height) // 2
        xnect_input_size = (2048, 2048)

        if width > height:
            xnect_input_image = cv2.resize(image[:, offset:width-offset, :], xnect_input_size)
        elif height > width:
            xnect_input_image = cv2.resize(image[offset:height-offset, :], xnect_input_size)
        else:
            xnect_input_image = image.copy()

        # Use XNect to detect any people in the image.
        self.__xnect.process_image(xnect_input_image)

        # Make the actual skeletons, and also the output visualisation if requested.
        skeletons = []                # type: List[Skeleton]
        visualisation = image.copy()  # type: np.ndarray

        # For each person index:
        for person_id in range(self.__xnect.get_num_of_people()):
            # If a person was detected with this index:
            if self.__xnect.is_person_active(person_id):
                # Make a skeleton for the person and add it to the list.
                skeleton_keypoints = {}  # type: Dict[str, Skeleton.Keypoint]

                # For each joint (ignoring the feet, as in the sample code, as they can be unstable):
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
            endpoint_a = self.__get_joint_pos_2d(person_id, joint_id, image.shape)   # type: Tuple[int, int]
            endpoint_b = self.__get_joint_pos_2d(person_id, parent_id, image.shape)  # type: Tuple[int, int]

            # If the endpoints are out of range, skip this bone.
            # FIXME: This is based on the sample code, but I don't get why the lower bounds are tested
            #        and the upper bounds aren't. This seems worth checking.
            if endpoint_a[0] <= 0 or endpoint_a[1] <= 0 or endpoint_b[0] <= 0 or endpoint_b[1] <= 0:
                continue

            # Draw the bone.
            colour = self.__get_person_colour(person_id)  # type: List[int]
            cv2.line(image, endpoint_a, endpoint_b, colour, 4)

    def __draw_joints(self, image: np.ndarray, person_id: int) -> None:
        """
        Draw the joints for the specified person onto an image the shape of the input image.

        :param image:       The image onto which to draw the joints (must have the same shape as the input image).
        :param person_id:   The index of the prson whose joints are to be drawn.
        """
        # Specify the parameters to pass to cv2.circle when drawing the joints.
        radius = 6
        thickness = -1

        # For each joint (ignoring the feet, as in the sample code, as they can be unstable):
        for joint_id in range(self.__xnect.get_num_of_3d_joints() - 2):
            # Compute the 2D position of the joint.
            pos = self.__get_joint_pos_2d(person_id, joint_id, image.shape)  # type: Tuple[int, int]

            # Draw the joint.
            colour = self.__get_person_colour(person_id)  # type: List[int]
            cv2.circle(image, pos, radius, colour, thickness)

    def __get_joint_pos_2d(self, person_id: int, joint_id: int, input_shape: tuple) -> Tuple[int, int]:
        """
        Get the position of the specified joint for the specified person in an image the shape of the input image.

        :param person_id:   The index of a person.
        :param joint_id:    The index of a joint.
        :param input_shape: The shape of the original input image.
        :return:            The position of the specified joint for the specified person in an image the shape
                            of the original input image.
        """
        return SkeletonDetector.__map_output_pos_to_input_pos(
            self.__xnect.project_with_intrinsics(self.__xnect.get_joint3d_ik(person_id, joint_id)), input_shape
        )

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
    def __map_output_pos_to_input_pos(pos: np.ndarray, input_shape: tuple) -> Tuple[int, int]:
        """
        Map a 2D joint position output by XNect to its equivalent position in the original input image.

        .. note::
            The input image was of size w x h. Without loss of generality, suppose we had w >= h, which is common.
            To get the XNect input, we centre-cropped the w x h image to make an image of size h x h, then scaled
            it up to 2048 x 2048. XNect outputs joints as positions (ox,oy) in a 1024 x 1024 image corresponding to
            the 2048 x 2048 input it was given. To work out where these would have been in the original input image,
            we must first scale these down to make them positions in the h x h image, then add an offset of (w-h)/2
            to the x coordinate. We thus get (ox * h / 1024 + (w-h)/2, oy * h / 1024), as shown in the code. Note
            that if we have h > w, we can simply do this the other way round.

        :param pos:             The joint position as output by XNect.
        :param input_shape:     The shape of the original input image.
        :return:                The equivalent joint position in the original input image.
        """
        height, width = input_shape[:2]
        offset = abs(width - height) // 2

        if width >= height:
            x = pos[0] * height / 1024 + offset
            y = pos[1] * height / 1024
        else:
            x = pos[0] * width / 1024
            y = pos[1] * width / 1024 + offset

        return tuple(np.round((x, y)).astype(int))
