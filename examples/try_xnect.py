import cv2
import numpy as np
import os

from smg.pyxnect import *


def draw_bones(img: np.ndarray, xnect: XNect, person: int) -> None:
    num_of_joints = xnect.get_num_of_3d_joints() - 2  # don't render feet, can be unstable

    for i in range(num_of_joints):
        parent_id = xnect.get_joint3d_parent(i)
        if parent_id == -1:
            continue

        # lookup 2 connected body/hand parts
        part_a = xnect.project_with_intrinsics(xnect.get_joint3d_ik(person, i))
        part_b = xnect.project_with_intrinsics(xnect.get_joint3d_ik(person, parent_id))

        if part_a[0] <= 0 or part_a[1] <= 0 or part_b[0] <= 0 or part_b[1] <= 0:
            continue

        colour = [int(_) for _ in xnect.get_person_colour(person)]
        cv2.line(img, tuple(np.round(part_a).astype(int)), tuple(np.round(part_b).astype(int)), colour, 4)


def draw_joints(img: np.ndarray, xnect: XNect, person: int) -> None:
    num_of_joints = xnect.get_num_of_3d_joints() - 2  # don't render feet, can be unstable

    radius = 6
    thickness = -1

    for i in range(num_of_joints):
        point2d = xnect.project_with_intrinsics(xnect.get_joint3d_ik(person, i))
        colour = [int(_) for _ in xnect.get_person_colour(person)]
        cv2.circle(img, tuple(np.round(point2d).astype(int)), radius, colour, thickness)


def main() -> None:
    # Change to the XNect executable directory. The XNect guys hard-coded the paths for some reason (argh!).
    os.chdir("D:/xnect/bin/Release")

    xnect = XNect()

    for i in range(50):
        img = cv2.imread("C:/smglib/smg-mapping/output-skeleton2/frame-{:06d}.color.png".format(i), cv2.IMREAD_UNCHANGED)

        xnect.process_image(img)
        img = cv2.resize(img, (1024, 1024))

        interesting = False

        for p in range(xnect.get_num_of_people()):
            if xnect.is_person_active(p):
                draw_bones(img, xnect, p)
                draw_joints(img, xnect, p)
                interesting = True

        cv2.imshow("Image", img)
        if interesting:
            cv2.waitKey()
        else:
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
