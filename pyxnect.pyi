import numpy as np


# CLASSES

class XNect:
    def __init__(self, config_file: str = "../../data/FullBodyTracker/"): ...
    def get_joint3d_ik(self, person: int, joint: int) -> np.ndarray: ...
    def get_joint3d_parent(self, joint: int) -> int: ...
    def get_num_of_3d_joints(self) -> int: ...
    def get_num_of_people(self) -> int: ...
    def get_person_colour(self, p: int) -> np.ndarray: ...
    def is_person_active(self, p: int) -> bool: ...
    def process_image(self, img: np.ndarray) -> None: ...
    def project_with_intrinsics(self, point: np.ndarray) -> np.ndarray: ...
