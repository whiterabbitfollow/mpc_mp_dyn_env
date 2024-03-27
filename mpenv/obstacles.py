import copy
from typing import List

import numpy as np
import trimesh

from mpenv.robot import rot_trans_to_SE3, SE3_inv, rot_z_to_SO3


def compute_rot_matrices(v_x):
    us_x = v_x
    us_x = (us_x.T / np.linalg.norm(us_x, axis=1)).T
    I = np.eye(3)
    v_z = I[:, 2]
    us_y = np.cross(v_z, us_x)
    us_y = (us_y.T / np.linalg.norm(us_y, axis=1)).T
    us_z = np.cross(us_x, us_y)
    us_z = (us_z.T / np.linalg.norm(us_z, axis=1)).T
    return np.concatenate([us_x.T[None], us_y.T[None], us_z.T[None]], axis=0)


def compute_rot_matrix_from_velocity(velocity):
    velocity = velocity.reshape(-1)
    direction_x = velocity
    direction_x = direction_x / np.linalg.norm(direction_x)
    e3 = np.eye(3)[:, 2]
    direction_y = np.cross(e3, direction_x)
    direction_z = np.cross(direction_x, direction_y)
    R = np.hstack([direction_x.reshape(-1, 1), direction_y.reshape(-1, 1), direction_z.reshape(-1, 1)])
    return R


class ObstacleStatic:

    def __init__(self, mesh, transform, dimensions, name=""):
        self.mesh = mesh
        self.name = name
        self.transform = transform
        self.dimensions = dimensions


class ObstacleDynamic:

    def __init__(self, dimensions, mesh, positions, velocities, via_pv_points, name=""):
        self.dimensions = dimensions
        self.mesh = mesh
        self.positions = positions
        if velocities.shape[1] == 2:
            velocities = np.hstack([velocities, np.zeros((velocities.shape[0], 1))])
        self.velocities = velocities
        self.rotations = compute_rot_matrices(velocities)
        self.name = name
        self.via_pv_points = via_pv_points

    def get_transform_at_time_step(self, ts):
        nr_positions = self.positions.shape[0]
        ts = ts % nr_positions  # TODO: need to interpolate between time steps
        i = int(ts)
        frac = ts - i
        if np.isclose(frac, 0):
            p = self.positions[i, :]
            R = self.rotations[:, :, i].T
        else:
            i_nxt = (i + 1) % nr_positions
            p = self.positions[i, :] * (1-frac) + self.positions[i_nxt, :] * frac
            v = self.velocities[i, :] * (1-frac) + self.velocities[i_nxt, :] * frac
            R = compute_rot_matrix_from_velocity(v)
        return rot_trans_to_SE3(p=p, R=R)


class ObstacleManager:

    def __init__(
            self,
            mesh_safe_region=None,
            delta_t=1,
            max_nr_static_obstacles=0,
            max_nr_dynamic_obstacles=10,
            max_nr_dynamic_obstacles_tries=10,
            max_nr_static_obstacles_tries_pre=40,
            max_nr_static_obstacles_tries_post=40,
            sampling_radius_dynamic_obstacles=0.5,
            static_positions_bounds=(-0.3, 0.3),
            nr_points_between_via_points=100
    ):
        self.static_positions_bounds = static_positions_bounds
        self.mesh_safe_region = mesh_safe_region
        self.sampling_radius_dynamic_obstacles = sampling_radius_dynamic_obstacles
        self.max_nr_dynamic_obstacles = max_nr_dynamic_obstacles
        self.max_nr_static_obstacles = max_nr_static_obstacles
        self.max_nr_dynamic_obstacles_tries = max_nr_dynamic_obstacles_tries
        self.max_nr_static_obstacles_tries_pre = max_nr_static_obstacles_tries_pre
        self.max_nr_static_obstacles_tries_post = max_nr_static_obstacles_tries_post
        self.collision_manager = trimesh.collision.CollisionManager()
        self.dynamic_obstacles: List[ObstacleDynamic] = []
        self.static_obstacles: List[ObstacleStatic] = []
        self.delta_t = delta_t
        self.nr_points_between_via_points = nr_points_between_via_points

    # def get_state_dict(self):
    #     return {
    #         "dynamic_obstacles": copy.deepcopy(self.dynamic_obstacles),
    #         "static_obstacles": copy.deepcopy(self.static_obstacles)
    #     }
    #
    # def load_state_dict(self, state):
    #     self.clear()
    #     self.dynamic_obstacles = state["dynamic_obstacles"]
    #     for o in self.dynamic_obstacles:
    #         self.collision_manager.add_object(
    #             name=o.name, mesh=o.mesh
    #         )
    #     self.static_obstacles = state["static_obstacles"]
    #     for o in self.static_obstacles:
    #         self.collision_manager.add_object(
    #             name=o.name, mesh=o.mesh
    #         )

    def __getstate__(self):
        self.clear_collision_manager()
        self.collision_manager = None
        state = copy.deepcopy(self.__dict__)
        self.collision_manager = trimesh.collision.CollisionManager()
        self.reset_collision_manager_from_obstacles()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.collision_manager = trimesh.collision.CollisionManager()
        self.reset_collision_manager_from_obstacles()

    def reset_collision_manager_from_obstacles(self):
        for o in self.dynamic_obstacles + self.static_obstacles:
            self.collision_manager.add_object(
                name=o.name, mesh=o.mesh
            )

    def clear_collision_manager(self):
        for o in self.dynamic_obstacles + self.static_obstacles:
            # TODO: should bool static objects..
            self.collision_manager.remove_object(o.name)

    def clear_dynamic_obstacles(self):
        for o in self.dynamic_obstacles:
            self.collision_manager.remove_object(o.name)

    def add_dynamic_obstacles(self):
        for o in self.dynamic_obstacles:
            self.collision_manager.add_object(
                name=o.name, mesh=o.mesh
            )

    def reset(self, seed=None):
        random_state = None
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
        self.clear()
        # TODO: make obstacle generator part of OM?
        # delta_t arbitrary?
        if self.max_nr_static_obstacles > 0:
            self.create_static_obstacles(nr_tries=self.max_nr_static_obstacles_tries_pre)
        if self.mesh_safe_region is not None:
            self.collision_manager.add_object(mesh=self.mesh_safe_region, name="safety_region")
        self.create_dynamic_obstacles()
        if self.max_nr_static_obstacles > 0:
            self.create_static_obstacles(nr_tries=self.max_nr_static_obstacles_tries_post)
        if self.mesh_safe_region is not None:
            self.collision_manager.remove_object(name="safety_region")
        if random_state is not None:
            np.random.set_state(random_state)

    def clear(self):
        self.clear_collision_manager()
        self.dynamic_obstacles.clear()
        self.static_obstacles.clear()

    def create_dynamic_obstacles(self):
        dynamic_obstacle_generator = DynamicObstacleGenerator(
            delta_t=self.delta_t,
            obstacle_manager=self,
            sampling_radius=self.sampling_radius_dynamic_obstacles,
            nr_points_between_via_points=self.nr_points_between_via_points
        )
        tries_cnt = 0
        max_tries_cnt = self.max_nr_dynamic_obstacles_tries
        while tries_cnt < max_tries_cnt and len(self.dynamic_obstacles) < self.max_nr_dynamic_obstacles:
            dynamic_obstacle = dynamic_obstacle_generator.generate_dynamic_obstacle()
            tries_cnt += 1
            if dynamic_obstacle is not None:
                self.append_dynamic_obstacle(dynamic_obstacle)

    def create_static_obstacles(self, nr_tries=0):
        meshes_static = {}
        collision_manager_static = trimesh.collision.CollisionManager()
        p_lower, p_upper = self.static_positions_bounds
        for cnt in range(nr_tries):
            p = np.random.uniform(p_lower, p_upper, size=3)
            p[-1] = 0
            angle = np.random.uniform(-np.pi, np.pi)
            R = rot_z_to_SO3(angle)
            dim_h = np.random.uniform(.05, .1)
            dim_w = np.random.uniform(.1, 1)
            dims = (dim_h, dim_w, 0.1)
            T = rot_trans_to_SE3(R, p)
            mesh = trimesh.creation.box(dims)
            mesh.apply_transform(T)
            name = f"{cnt}"
            # if not collision_manager_static.in_collision_single(mesh):
            collision_manager_static.add_object(name, mesh)
            meshes_static[name] = (mesh, T, dims)
        if self.dynamic_obstacles:
            ts_end = self.dynamic_obstacles[0].positions.shape[0]
            for ts in range(ts_end):
                self.set_time(ts)
                collision, names = self.collision_manager.in_collision_other(
                    collision_manager_static, return_names=True
                )
                for _, static_name in names:
                    if static_name in meshes_static:
                        collision_manager_static.remove_object(static_name)
                        meshes_static.pop(static_name)
        elif self.mesh_safe_region is not None:
            collision, names = collision_manager_static.in_collision_single(
                self.mesh_safe_region, return_names=True
            )
            for static_name in names:
                if static_name in meshes_static:
                    collision_manager_static.remove_object(static_name)
                    meshes_static.pop(static_name)
        for name in list(meshes_static.keys())[:self.max_nr_static_obstacles]:
            self.append_static_obstacle(ObstacleStatic(*meshes_static[name]))

    def append_dynamic_obstacle(self, obstacle: ObstacleDynamic):
        name = obstacle.name or f"obstacle_dynamic_{len(self.dynamic_obstacles)}"
        self.collision_manager.add_object(
            name=name, mesh=obstacle.mesh
        )
        self.dynamic_obstacles.append(obstacle)
        if not obstacle.name:
            obstacle.name = name

    def append_static_obstacle(self, obstacle: ObstacleStatic):
        name = obstacle.name or f"obstacle_static_{len(self.static_obstacles)}"
        self.collision_manager.add_object(name=name, mesh=obstacle.mesh)
        self.static_obstacles.append(obstacle)
        if not obstacle.name:
            obstacle.name = name

    def is_collision_free_time_step(self, ts, mesh):
        self.set_time(ts)
        return not self.collision_manager.in_collision_single(mesh)

    def set_time(self, ts):
        for i, o in enumerate(self.dynamic_obstacles):
            T = o.get_transform_at_time_step(ts)
            # if o.name in self.collision_manager._objs: # debug stuff
            self.collision_manager.set_transform(name=o.name, transform=T)

    def clone_collision_manager_with_dynamic_obstacles(self, ts):
        cm = trimesh.collision.CollisionManager()
        for i, o in enumerate(self.dynamic_obstacles):
            T = o.get_transform_at_time_step(ts)
            # if o.name in self.collision_manager._objs: # debug stuff
            cm.add_object(mesh=o.mesh, name=o.name, transform=T)
        return cm


class DynamicObstacleGenerator:

    def __init__(
            self,
            delta_t,
            obstacle_manager,
            sampling_radius=0.5,
            nr_points_between_via_points=100
    ):
        self.obstacle_manager = obstacle_manager
        self.delta_t = delta_t
        self.nr_points_between_via_points = nr_points_between_via_points
        self.r = sampling_radius
        self.dim = 2

    def generate_dynamic_obstacle(self, nr_via_points=3):
        # r_c = 0.1
        # h_c = 0.5
        # mesh = trimesh.creation.cylinder(radius=r_c, height=h_c)
        # width, height, depth = np.random.uniform(.1, .3, size=3)
        radius = 0.1
        dimensions = (radius, )
        # mesh = trimesh.creation.box(dimensions)
        mesh = trimesh.creation.cylinder(radius, .05)
        trajectory = np.array([]).reshape((0, self.dim * 2))
        ts = 0
        via_pv_points = self.sample_via_pv_point().reshape(1, -1)
        for i in range(nr_via_points):
            if i == nr_via_points - 1:
                trajectory_segment = self.compute_segment(pv_start=via_pv_points[i], pv_end=via_pv_points[0])
                is_collision_free = self.is_segment_collision_free(mesh, trajectory_segment, ts_start=ts)
                results = (via_pv_points[0], trajectory_segment)
                if not is_collision_free:
                    # print("Couldn't connect end...")
                    # TODO: think of this...
                    return None
            else:
                results = self.generate_collision_free_segment(mesh, ts, via_pv_points[i])
            if results is not None:
                pv_end, trajectory_segment = results
                ts += trajectory_segment.shape[0]
                # trajectory_segment[-1, :]
                via_pv_points = np.vstack([via_pv_points, pv_end])
                trajectory = np.vstack([trajectory, trajectory_segment[:-1, :]])
            else:
                # TODO: implement backup
                return None
        positions, velocities = trajectory[:, :self.dim], trajectory[:, self.dim:]
        return ObstacleDynamic(dimensions, mesh, positions, velocities, via_pv_points)

    @staticmethod
    def generate_position_within_ball(r, dim=2):
        point = np.ones((dim,)) * np.inf
        while np.linalg.norm(point) > r:
            point = np.random.uniform(-r, r, size=(dim,))
        return point

    @staticmethod
    def generate_position_within_annululs(r, dim=2):
        r_min = 0.1
        while True:
            point = np.random.uniform(-r, r, size=(dim,))
            dist = np.linalg.norm(point)
            if r_min < dist < r:
                return point

    def sample_position(self):
        return self.generate_position_within_ball(self.r, dim=self.dim)

    def sample_velocity(self):
        v_min, v_max = -1, 1
        return np.random.uniform(v_min, v_max, size=(self.dim, ))

    def sample_via_pv_point(self):
        return np.hstack([self.sample_position(), self.sample_velocity()])

    def generate_collision_free_segment(self, mesh, ts_start, pv_start):
        is_collision_free = False
        cnt = 0
        max_cnt = 10
        while not is_collision_free and cnt < max_cnt:
            pv_end = self.sample_via_pv_point()
            segment = self.compute_segment(pv_start, pv_end)
            is_collision_free = self.is_segment_collision_free(mesh, segment, ts_start)
            if is_collision_free:
                return (pv_end, segment)
            cnt += 1
        return None

    def compute_segment(self, pv_start, pv_end):
        nr_points_between_via_points = self.nr_points_between_via_points # np.random.randint(200, 300)    # 40
        coeffs = calculate_cubic_coefficients_between_via_points(
            position_start=pv_start[:self.dim],
            velocity_start=pv_start[self.dim:],
            position_end=pv_end[:self.dim],
            velocity_end=pv_end[self.dim:],
            delta_t=self.delta_t
        )
        positions, speeds, *_ = interpolate_trajectory_from_cubic_coeffs(
            *coeffs,
            delta_t=self.delta_t,
            nr_points=nr_points_between_via_points
        )
        return np.hstack([positions, speeds])

    def is_segment_collision_free(self, mesh, segment, ts_start):
        positions, speeds = segment[:, :self.dim], segment[:, self.dim:]
        if speeds.shape[1] == 2:
            speeds = np.hstack([speeds, np.zeros((speeds.shape[0], 1))])
        Rs = compute_rot_matrices(speeds)
        T_inv = np.eye(4)
        is_collision_free = True
        ts = ts_start
        for i in range(positions.shape[0]):
            p = positions[i, :]
            R = Rs[:, :, i].T
            T = rot_trans_to_SE3(p=p, R=R)
            mesh.apply_transform(T @ T_inv)
            T_inv = SE3_inv(T)
            is_collision_free = self.obstacle_manager.is_collision_free_time_step(ts + i, mesh)
            if not is_collision_free:
                # viz_collision(positions, T)
                break
        mesh.apply_transform(T_inv)
        return is_collision_free


def calculate_cubic_coefficients_between_via_points(
        position_start,
        velocity_start,
        position_end,
        velocity_end,
        delta_t
    ):
    beta_s = position_start
    beta_d_s = velocity_start
    beta_e = position_end
    beta_d_e = velocity_end
    a_0 = beta_s
    a_1 = beta_d_s
    a_2 = 3 * beta_e - 3 * beta_s - 2 * beta_d_s * delta_t - beta_d_e * delta_t
    a_2 /= (delta_t ** 2)
    a_3 = 2 * beta_s + (beta_d_s + beta_d_e) * delta_t - 2 * beta_e
    a_3 /= (delta_t ** 3)
    return a_0, a_1, a_2, a_3


def interpolate_trajectory_from_cubic_coeffs(a_0, a_1, a_2, a_3, delta_t, nr_points):
    ts = np.linspace(0, delta_t, nr_points)
    poses, spds, accs, jrks = [], [], [], []
    for t in ts:
        pos, spd, acc, jrk = calculate_pose_speed_acc_jerk_from_cubic_coefficients(a_0, a_1, a_2, a_3, t)
        poses.append(pos)
        spds.append(spd)
        accs.append(acc)
        jrks.append(jrk)
    return np.c_[poses], np.c_[spds], np.c_[accs], np.c_[jrks]


def calculate_pose_speed_acc_jerk_from_cubic_coefficients(a_0, a_1, a_2, a_3, t):
    pos = a_0 + a_1 * t + a_2 * t ** 2 + a_3 * t ** 3
    spd = a_1 + 2 * a_2 * t + 3 * a_3 * t ** 2
    acc = 2 * a_2 + 6 * a_3 * t
    jrk = 6 * a_3
    return pos, spd, acc, jrk