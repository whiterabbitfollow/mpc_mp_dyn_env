import trimesh
import numpy as np

class MobileRobot:

    def __init__(self, radius=0.1, height=0.1, max_actuation=0.1, **kwargs):
        # self.config = np.zeros((2, ))
        self.T = np.eye(4)
        # self.T[:2, -1] = self.config
        self.config = self.T[:2, -1]
        self.max_actuation = max_actuation

        self.body_radius = radius
        self.body_height = height
        self.body_mesh = trimesh.creation.cylinder(radius=self.body_radius, height=self.body_height)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object("body", self.body_mesh)

    def set_config(self, config):
        self.config[:2] = config[:2]
        self.collision_manager.set_transform("body", self.T)


def angle_from_SE3_rot_z(T):
    return np.arctan2(T[1, 0], T[0, 0])


def rot_trans_to_SE3(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        p = p.ravel()
        T[:p.shape[0], 3] = p
    return T


def SE3_mul(T, p):
    return (T[:3, :3] @ p.reshape(-1, 1) + T[:3, 3].reshape(-1, 1)).ravel()  # TODO: this is dangerous


def SE3_inv(T):
    R = T[:3, :3]
    p = T[:3, 3].reshape(-1, 1)
    return np.block(
        [
            [R.T, -R.T @ p],
            [T[-1, :].reshape(1, -1)]
        ]
    )


def rot_z_to_SO3(angle):
    R = np.eye(3)
    R[0, 0] = np.cos(angle)
    R[1, 1] = np.cos(angle)
    R[0, 1] = -np.sin(angle)
    R[1, 0] = np.sin(angle)
    return R


if __name__ == "__main__":
    robot = MobileRobot()
    robot.set_config(np.array([0.3, 0.3]))
