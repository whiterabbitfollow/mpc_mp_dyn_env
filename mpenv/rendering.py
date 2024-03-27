import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from mpenv.robot import angle_from_SE3_rot_z, SE3_mul



class RenderEngineWorld2D:

    def __init__(self, world):
        self.world = world

    def render_world(
            self,
            ax,
            t=None,
            config=None,
            config_goal=None,
            dynamic_obstacles=True,
            static_obstacles=True
    ):
        robot = self.world.robot
        obstacles = self.world.obstacles
        if t is not None:
            self.world.set_time(t)
        t = self.world.t
        config = config if config is not None else robot.config
        # render_robot_configuration_meshes(ax, self.robot, color="orange", alpha=0.5)
        # robot.set_config(config_end)
        # render_robot_configuration_meshes(ax, robot, color="blue", alpha=0.5)
        # robot.set_config(config)
        collision, names = robot.collision_manager.in_collision_other(
            obstacles.collision_manager, return_names=True
        )
        color = "red" if collision else "blue"
        config = tuple(config)
        ax.add_patch(Circle(tuple(config), radius=robot.body_radius, color=color, alpha=.5, label="Robot"))
        if config_goal is not None:
            ax.add_patch(Circle(tuple(config_goal), radius=robot.body_radius, color="red", alpha=0.3, label=f"Goal region"))
            # ax.scatter(config_goal[0], config_goal[1], color="red", label=f"Goal position")
        obstacles_in_collision = [obstacle_name for _, obstacle_name in names]
        self.render_obstacle_manager_objects(
            ax,
            obstacles,
            self.world.t,
            obstacles_in_collision,
            dynamic_obstacles=dynamic_obstacles,
            static_obstacles=static_obstacles
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(f"Time {t:0.2f}")

    def animate(self):
        fig = plt.figure(1)
        ax = fig.add_subplot()
        dt = 1.0
        t = 0
        robot = self.world.robot
        obstacles = self.world.obstacles
        config_start = np.random.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1])
        config_goal = np.random.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1])
        for i in range(1_000):
            self.world.set_time(t)
            # config = self.robot.get_config()
            self.render_world(ax, t=t, config=config_start, config_goal=config_goal)
            plt.pause(0.01)
            ax.cla()
            t += dt
        plt.close()

    def render_obstacle_manager_objects(
            self,
            ax,
            om,
            ts,
            obstacles_in_collision=None,
            static_obstacles=True,
            dynamic_obstacles=False
    ):
        obstacles_in_collision = obstacles_in_collision or []
        if static_obstacles:
            for o in om.static_obstacles:
                if o.name in obstacles_in_collision:
                    color = "red"
                    alpha = 1.0
                else:
                    color = "green"
                    alpha = 0.5
                width, height, _ = o.dimensions
                T = o.transform
                box_corner_lower_local = np.array([-width / 2, -height / 2, 0])
                x, y, _ = SE3_mul(T, box_corner_lower_local)
                # x, y = p_x, p_y
                angle = np.rad2deg(angle_from_SE3_rot_z(T))
                # angle = 0
                ax.add_patch(Rectangle((x, y), width, height, angle=angle, color=color))
        if dynamic_obstacles:
            for i, o in enumerate(om.dynamic_obstacles):
                ax.plot(o.positions[:, 0], o.positions[:, 1], marker=".", label=f"Obstacle {i + 1} path")
            for i, o in enumerate(om.dynamic_obstacles):
                radius, = o.dimensions
                T = o.get_transform_at_time_step(ts)
                x, y = T[:2, -1]
                # box_corner_lower_local = np.array([-width / 2, -height / 2, 0])
                # x, y, _ = SE3_mul(T, box_corner_lower_local)
                # angle = np.rad2deg(angle_from_SE3_rot_z(T))
                # # angle = 0
                if o.name in obstacles_in_collision:
                    color = "red"
                    alpha = 1.0
                else:
                    color = "darkgreen"
                    alpha = 0.5
                label = "Obstacle" if i == 0 else None
                ax.add_patch(Circle((x, y), radius, color=color, label=label))
                # ax.add_patch(Rectangle((x, y), width, height, angle, color=color, label=label))
                # ax.scatter(p_x, p_y)
                # render_mesh(ax, m, color=color, alpha=alpha)
                # # return PatchCollection(rectangles, **kwargs)
                #
                #
                R = T[:2, :2]
                p = T[:2, 3]
                for e, c in zip(R.T, ["red", "green"]):
                    line = np.vstack([p, p + e * 0.1])
                    ax.plot(line[:, 0], line[:, 1], c=c)
