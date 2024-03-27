import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from mpenv.obstacles import ObstacleManager
from mpenv.robot import MobileRobot


@dataclass
class Range:
    lower: float
    upper: float

    def to_tuple(self):
        return self.lower, self.upper


@dataclass
class WorldData2D:
    x: Range
    y: Range

    def __init__(self, xs, ys):
        self.x = Range(*xs)
        self.y = Range(*ys)


class MobileMovingBoxesWorld:

    def __init__(
            self,
            robot_data=None,
            kwargs_obstacles=None,
            safe_region_radius=0.2
    ):
        data = WorldData2D((-1, 1), (-1, 1))
        robot = MobileRobot(
            max_actuation=0.1, **(robot_data or {})
        )
        self.robot = robot
        mesh_safe_region = None     # trimesh.creation.icosphere(radius=safe_region_radius)
        self.data = data
        self.world_dim = 2
        self.config_start = None # np.zeros((self.world_dim,))
        # np.random.uniform(self.world_limits[:, 0], self.world_limits[:, 1])
        self.world_limits = np.vstack([self.data.x.to_tuple(), self.data.y.to_tuple()])
        self.obstacles = ObstacleManager(
            mesh_safe_region=mesh_safe_region,
            **(kwargs_obstacles or {})
        )

    def sample_feasible_config(self):
        return np.random.uniform(self.world_limits[:, 0], self.world_limits[:, 1])

    def sample_collision_free_config_at_time(self, t):
        self.set_time(t)
        while True:
            q = np.random.uniform(self.world_limits[:, 0], self.world_limits[:, 1])
            coll_free = self.is_collision_free_config(q)
            distance = self.get_distance()
            if coll_free and distance > 0.025:
                break
        return q

    def sample_collision_free_config_during_time_window(self, t_0=0, dt=10):
        tries_max = 1000
        try_cnt = 0
        while try_cnt < tries_max:
            q = np.random.uniform(self.world_limits[:, 0], self.world_limits[:, 1])
            cnt = 0
            self.set_time(t_0)
            coll_free = self.is_collision_free_config(q)
            while coll_free and cnt < dt:
                cnt += 1
                self.set_time(t_0 + cnt)
                coll_free = self.is_collision_free_config(q)
            if coll_free:
                return q
            try_cnt += 1
        return None

    def set_time(self, t):
        self.t = t
        self.obstacles.set_time(t)

    def clear_dynamic_obstacles(self):
        self.obstacles.clear_dynamic_obstacles()

    def add_dynamic_obstacles(self):
        self.obstacles.add_dynamic_obstacles()
        self.obstacles.set_time(self.t)

    def reset(self, seed=None, reset_obstacles=True):
        if reset_obstacles:
            self.obstacles.reset(seed=seed)
        self.set_time(0)
        self.config_start = self.sample_collision_free_config_during_time_window(0, 10)
        assert self.config_start is not None
        self.robot.set_config(self.config_start)

    def save_state(self, file_name):
        with open(file_name, "wb") as fp:
            pickle.dump(self.obstacles, fp)

    def load_state(self, file_name):
        with open(file_name, "rb") as fp:
            self.obstacles = pickle.load(fp)
        self.set_time(0)

    def is_state_collision_free(self, state):
        config = state[:-1]
        t = state[-1]
        self.set_time(t)
        self.robot.set_config(config)
        return not self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)

    def get_distance(self):
        return self.robot.collision_manager.min_distance_other(self.obstacles.collision_manager)

    def is_collision_free_config(self, config):
        self.robot.set_config(config)
        return not self.robot.collision_manager.in_collision_other(self.obstacles.collision_manager)

    def is_collision_free_transition(self, x_src, x_dst, nr_coll_steps=2, include_src=False, return_distance=False):
        is_collision_free = False
        i_start = 0 if include_src else 1
        distance = np.inf
        for i in range(i_start, nr_coll_steps + 1):
            alpha = i / nr_coll_steps
            state = x_dst * alpha + (1 - alpha) * x_src
            is_collision_free = self.is_state_collision_free(state)
            if return_distance:
                distance = min(self.get_distance(), distance)
            if not is_collision_free:
                break
        if return_distance:
            return distance, is_collision_free
        else:
            return is_collision_free

    def get_smallest_distance_in_transition(self, x_src, x_dst, nr_coll_steps=2, include_src=False):
        is_collision_free = False
        i_start = 0 if include_src else 1
        distance = np.inf
        for i in range(i_start, nr_coll_steps + 1):
            alpha = i / nr_coll_steps
            state = x_dst * alpha + (1 - alpha) * x_src
            is_collision_free = self.is_state_collision_free(state)
            distance = min(self.get_distance(), distance)
        return distance, is_collision_free

    def collision_check_transition(self, x_src, x_dst, nr_coll_steps=2, include_src=False):
        is_collision_free = False
        i_start = 0 if include_src else 1
        x_coll_free = x_src
        for i in range(i_start, nr_coll_steps + 1):
            alpha = i / nr_coll_steps
            x = x_dst * alpha + (1 - alpha) * x_src
            is_collision_free = self.is_state_collision_free(x)
            if not is_collision_free:
                break
            x_coll_free = x
        return is_collision_free, x_coll_free


if __name__ == "__main__":
    from mpenv.rendering import RenderEngineWorld2D
    world = MobileMovingBoxesWorld(
        kwargs_obstacles=dict(
            max_nr_static_obstacles=8,
            max_nr_dynamic_obstacles=2,
            max_nr_dynamic_obstacles_tries=40,
            max_nr_static_obstacles_tries=100,
            static_positions_bounds=(-0.5, 0.5)
        )
    )
    world.reset(seed=5)
    render = RenderEngineWorld2D(world)
    fig, ax1 = plt.subplots(1, 1)
    for t in range(300):
        world.robot.set_config(np.array([0.2, 0]))
        world.set_time(t)
        render.render_world(ax1)
        ax1.axis("equal")
        plt.pause(0.1)
        ax1.cla()

    #plt.show()
    # world.view()