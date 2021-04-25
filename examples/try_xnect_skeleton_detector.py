import cv2
import numpy as np

from smg.pyxnect import SkeletonDetector


def main() -> None:
    skeleton_detector = SkeletonDetector()

    for i in range(250):
        img = cv2.imread("C:/smglib/smg-mapping/output-xnect/frame-{:06d}.color.png".format(i), cv2.IMREAD_UNCHANGED)

        skeletons, visualisation = skeleton_detector.detect_skeletons(img, np.eye(4), visualise=True)

        cv2.imshow("Visualisation", visualisation)
        if skeletons is not None:
            cv2.waitKey(50)
        else:
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
