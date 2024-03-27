import numpy as np

from mpenv.rendering import RenderEngineWorld2D
from mpenv.world import MobileMovingBoxesWorld


class MPEnv:

    def __init__(
            self,
            kwargs_obstacles=None,
            t_max=500,
            kwargs_world=None
    ):
        kwargs_obstacles_default = dict(
            delta_t=1,
            max_nr_static_obstacles=20,
            max_nr_dynamic_obstacles=3,
            max_nr_dynamic_obstacles_tries=40,
            max_nr_static_obstacles_tries_pre=3,
            max_nr_static_obstacles_tries_post=197,
            static_positions_bounds=(-1, 1),
            sampling_radius_dynamic_obstacles=0.9,
            nr_points_between_via_points=25
        )
        self.world = MobileMovingBoxesWorld(
            kwargs_obstacles=(kwargs_obstacles or kwargs_obstacles_default),
            **(kwargs_world or {})
        )
        self.render_engine = RenderEngineWorld2D(self.world)
        self.action_max = self.world.robot.max_actuation
        self.config_dim = self.world.world_dim
        W = 2
        self.action_dim = self.config_dim
        self.x = np.zeros((W * 2, 1))
        self.config_goal = None
        self.t = 0
        self.t_max = t_max
        self.fig_ax = None

    def reset(
            self,
            seed=None,
            hard=False,
            config=None,
            config_goal=None,
            persistent_dir=None,
            t_start=0
    ):
        self.t = t_start
        self.world.set_time(self.t)
        if hard:
            if persistent_dir is None:
                self.world.reset(seed)
            elif (persistent_dir / f"env_{seed}.pckl").exists():
                path_to_file = (persistent_dir / f"env_{seed}.pckl")
                self.load_state(path_to_file)
            else:
                self.world.reset(seed)
                self.save_state(persistent_dir / f"env_{seed}.pckl")
        else:
            self.world.reset(seed=seed, reset_obstacles=False)
        config = config if config is not None else self.world.config_start.copy()
        velocity_start = np.zeros((2, ))
        self.x = np.hstack([config, velocity_start])[None].T
        if config_goal is not None:
            self.config_goal = config_goal
        else:
            while True:
                config_goal = self.world.sample_collision_free_config_at_time(0)
                if np.linalg.norm(config_goal - self.x[:2]) > 0.4 and np.linalg.norm(config_goal) < 0.7:
                    break
            self.config_goal = config_goal
        return self.x.copy(), self.config_goal.copy()

    def set_env_at_time(self, t):
        self.t = t
        self.world.set_time(self.t)

    def step(self, action: np.ndarray):
        W = 2
        I_2 = np.eye(W)
        O_2 = np.zeros((W, W))
        A = np.block(
            [
                [I_2, I_2],
                [O_2, I_2]]
        )
        B = np.block([
            [I_2 * .5],
            [I_2]
        ])
        if action.size != self.action_dim:
            raise RuntimeError(f"Bad action dimension, got {action.size}, need {self.action_dim}")
        limits = self.world.world_limits
        u = action.ravel()[None].T
        x_nxt = A @ self.x + B @ u
        config = self.x[:2]
        qt_src = np.append(config, self.t)
        q_nxt = x_nxt[:2].ravel()
        qt_dst = np.append(q_nxt, self.t + 1)
        coll_free, qt_nxt = self.world.collision_check_transition(qt_src, qt_dst)
        if not ((limits[:, 0] <= q_nxt) & (q_nxt <= limits[:, 1])).all():
            q_nxt = np.clip(q_nxt, limits[:, 0], limits[:, 1])
            self.x[:2, 0] = q_nxt
            self.x[2:, 0] = 0
        else:
            self.x = x_nxt
        self.t = self.t + 1
        self.world.set_time(self.t)
        obs = (self.x.copy(), self.config_goal.copy())
        return obs, coll_free, self.t > self.t_max

    def render(self, title="", pause_time=0.1):
        if self.fig_ax is None:
            self.fig_ax = plt.subplots()
        fig, ax = self.fig_ax
        self.render_engine.render_world(ax, config=self.x[:2], config_goal=self.config_goal)
        if title:
            fig.suptitle(title)
        plt.pause(pause_time)
        ax.cla()

    def view(self):
        fig, ax = plt.subplots()
        self.render_engine.render_world(ax, config=self.x[:2], config_goal=self.config_goal)
        plt.show()

    def save_state(self, file_name):
        self.world.save_state(file_name)

    def load_state(self, file_name):
        self.world.load_state(file_name)
        self.reset()

    def set_state(self, x):
        self.x = x.copy()
        self.set_config(self.x[:2].ravel())

    def set_config(self, config):
        self.x[:2, 0] = config
        self.world.robot.set_config(config)


if __name__ == "__main__":
    from collections import Counter
    import matplotlib.pyplot as plt
    import numpy as np
    from mpenv.environment import MPEnv

    env_kwargs = dict(kwargs_obstacles=dict(
        delta_t=1,
        max_nr_static_obstacles=10,
        max_nr_dynamic_obstacles=3,
        max_nr_dynamic_obstacles_tries=40,
        max_nr_static_obstacles_tries_pre=3,
        max_nr_static_obstacles_tries_post=197,
        static_positions_bounds=(-1, 1),
        sampling_radius_dynamic_obstacles=0.9,
        nr_points_between_via_points=25
    )
    )
    env = MPEnv(
        **env_kwargs
    )
    env.reset(seed=0, hard=True)
    env.view()