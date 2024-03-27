from collections import defaultdict

import numpy as np
import trimesh


def get_transition(qt_src, qt_dst, delta_min=1e-2):
    delta = np.linalg.norm(qt_src - qt_dst)
    nr_points = int(np.ceil(delta / delta_min))
    betas = np.linspace(0, 1, nr_points)
    line_points = qt_src.reshape(-1, 1) * (1 - betas) + qt_dst.reshape(-1, 1) * betas
    line_points = line_points.T
    return line_points


class CollisionChecker:

    def __init__(
            self,
            scl,
            ts_obs,
            ts_p,
            motion_model_poly_degree=3
    ):
        self.ts_obs = ts_obs
        self.ts_p = ts_p
        self.radius_robot = 0.1
        self.cm_robot = trimesh.collision.CollisionManager()
        self.mesh = trimesh.creation.cylinder(radius=self.radius_robot, height=0.05)
        self.cm_robot.add_object("body", self.mesh)
        self.T = np.eye(4)
        self.scl = scl
        self.cm_robot.set_transform("body", self.T)
        self.cm_static = trimesh.collision.CollisionManager()
        self.Th = None  # TODO: need to scale...
        self.rs = None
        self.meshes = []
        self.motion_model_poly_degree = motion_model_poly_degree

    def get_estimated_motion(self):
        if self.Th is None:
            return None
        ts = np.arange((self.ts_obs - 1), (self.ts_obs - 1) + self.ts_p)
        T = ts.size
        X = np.vander(ts, N=self.motion_model_poly_degree)  # [T, 3]
        ps_objects = X @ self.Th  # [T, 3] [3, W * N_obs] -> [T, W * N_obs]
        ps_objects = ps_objects.reshape((T, -1, 2))
        return ps_objects

    def clear(self):
        for n, m in self.meshes:
            self.cm_static.remove_object(n)
        self.meshes.clear()
        self.clear_dynamic_obstacles()

    def clear_dynamic_obstacles(self):
        self.Th = None
        self.rs = None

    def add_static_obstacles(self, static_obstacles):
        for m in static_obstacles:
            self.cm_static.add_object(m.name, m.mesh)
            self.meshes.append((m.name, m.mesh))

    def is_state_collision_free_static(self, x):
        # TODO: need to check this
        self.T[:2, -1] = x[:2]
        self.cm_robot.set_transform("body", self.T)
        return not self.cm_robot.in_collision_other(self.cm_static)

    def is_state_collision_free(self, x):
        return (
                self.is_state_collision_free_static(x)
                and
                self.are_states_collision_free_dynamic(x.reshape((1, -1)))
        )

    def are_states_collision_free(self, xs):
        """
        xs [T, 3]
        """
        return (
                all(self.is_state_collision_free_static(x) for x in xs)
                and
                self.are_states_collision_free_dynamic(xs)
        )

    def are_states_collision_free_dynamic(self, xs):
        # Ts [3, N_obs] ??
        collision_free = True
        if self.Th is not None:
            N_obs = self.rs.shape[0]
            r = 0.1
            W = 2
            ts = (xs[:, -1] / self.scl)
            mask = ts < self.ts_p
            if mask.any():
                ps = xs[mask, :-1]
                ts = ts[mask] + (self.ts_obs - 1)
                X = np.vander(ts, N=self.motion_model_poly_degree)  # [T, 3]
                ps_objects = X @ self.Th  # [T, 3] [3, W * N_obs] -> [T, W * N_obs]
                direction = ps.reshape((-1, 1, 2)) - ps_objects.reshape((-1, N_obs, 2))  # [T, N_obs, W]
                dist = np.hypot(direction[..., 0], direction[..., 1])  # [T, N_obs]
                body_distance = dist - self.rs.reshape((1, -1)) - r  # [T, N_obs]
                collision_free = body_distance.min() > 0
        return collision_free

    def get_closest_points_static(self, ps):
        # trimesh.proximity.closest_point(self.meshes, ps)
        points_all = []
        for p in ps:
            self.T[:2, -1] = p
            self.cm_robot.set_transform("body", self.T)
            points = []
            for _, m in self.meshes:
                distance, names, data = self.cm_robot.min_distance_single(
                    m, return_data=True, return_name=True
                )
                points.append(data.point("__external")[:2])
            points_all.append(points)
        # [P, N, 2]
        points_all = np.stack(points_all)
        return points_all

    def get_static_obstacles_closest_points_and_normals(self, ps):
        boundary_points = self.get_closest_points_static(ps)
        boundary_normals = ps.reshape((-1, 1, 2)) - boundary_points
        boundary_normals = boundary_normals / np.linalg.norm(boundary_normals, axis=-1, keepdims=True)
        boundary_points = boundary_points + boundary_normals * 0.1
        return boundary_points, boundary_normals

    def get_closest_points_dynamic(self, xs):
        normals = np.array([]).reshape((xs.shape[0], 0, 2))
        points = normals
        if self.Th is not None:
            N_obs = self.rs.shape[0]
            # ts_pred = np.arange(T_obs-1, T_obs - 1 + T_pred)
            ts = xs[:, -1] + (self.ts_obs - 1)  # TODO: very buggy
            ps = xs[:, :-1]
            X = np.vander(ts, N=self.motion_model_poly_degree)  # [T, 3]
            ps_objects = X @ self.Th  # [T, 3] [3, W * N_obs] -> [T, W * N_obs]
            normals = ps.reshape((-1, 1, 2)) - ps_objects.reshape((-1, N_obs, 2))  # [T, N_obs, W]
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
            points = normals * (self.rs.reshape((1, -1, 1)) + self.radius_robot) + ps_objects.reshape((-1, N_obs, 2))
        return points, normals

    def get_dynamic_obstacles_closest_points_and_normals(self, xs):
        empty_array = np.array([]).reshape((0, 0, 2))
        ts = xs[:, -1]
        mask = ts < self.ts_p
        if mask.any():
            boundary_points, boundary_normals = self.get_closest_points_dynamic(xs[mask])
        else:
            boundary_points, boundary_normals = empty_array, empty_array
        return boundary_points, boundary_normals


class Tree:

    def __init__(self, size_max, dim):
        self.verts = np.zeros((size_max, dim))
        self.edges_parent = dict()
        self.cnt = 0
        self.cnt_max = size_max

    def clear(self):
        self.verts[:self.cnt] = 0
        self.cnt = 0
        self.edges_parent.clear()

    def add_vert_to_tree(self, v, v_parent=None):
        self.verts[self.cnt] = v
        self.edges_parent[tuple(v)] = tuple(v_parent) if v_parent is not None else None
        self.cnt += 1

    def get_nearest_reachable_neigh(self, qt):
        pass

    def get_vertices(self):
        return self.verts[:self.cnt]

    def get_path_from_vert(self, v):
        path = [v]
        while True:
            v_p = self.edges_parent[tuple(v)]
            if v_p is None:
                break
            else:
                path.append(v_p)
                v = v_p
        return np.vstack(path)


class TreeStart(Tree):

    def __init__(self, *args, speed_max, **kwargs):
        """
            funnel [time, centroid_xy, radius]
        """
        self.speed_max = speed_max
        super().__init__(*args, **kwargs)

    def get_nearest_reachable_neigh(self, qt):
        V = self.verts[:self.cnt]
        mask = V[:, -1] < qt[-1]
        qt_closest = None
        delta_closest = np.inf
        if mask.any():
            V_ = V[mask]
            dq = np.linalg.norm(V_[:, :-1] - qt[:-1], axis=-1)
            dt = qt[-1] - V_[:, -1]
            speed = dq / dt
            mask = speed < self.speed_max
            if mask.any():
                V__ = V_[mask]
                deltas = np.linalg.norm(V__ - qt, axis=-1)
                i_min = deltas.argmin()
                qt_closest = V__[i_min]
                delta_closest = deltas[i_min]
        return delta_closest, qt_closest


class TreeGoal(Tree):

    def __init__(self, *args, speed_max, **kwargs):
        self.speed_max = speed_max
        super().__init__(*args, **kwargs)
        self.cnt_ws = None

    def get_nearest_reachable_neigh(self, qt):
        V = self.verts[:self.cnt]
        mask = V[:, -1] > qt[-1]
        qt_closest = None
        delta_closest = np.inf
        if mask.any():
            V_ = V[mask]
            dq = np.linalg.norm(V_[:, :-1] - qt[:-1], axis=-1)
            dt = V_[:, -1] - qt[-1]
            speed = dq / dt
            mask = speed < self.speed_max
            if mask.any():
                V__ = V_[mask]
                deltas = np.linalg.norm(V__ - qt, axis=-1)
                i_min = deltas.argmin()
                qt_closest = V__[i_min]
                delta_closest = deltas[i_min]
        return delta_closest, qt_closest

    def prune_added_verticies(self):
        for v in self.verts[self.cnt_ws:self.cnt]:
            self.edges_parent.pop(tuple(v))
        self.verts[self.cnt_ws:self.cnt] = 0
        self.cnt = self.cnt_ws


class TreeStatic(Tree):

    def __init__(self, coll_checker, neigh_delta_max, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_2_go = defaultdict(lambda: np.inf)
        self.edges_children = defaultdict(list)
        self.coll_checker = coll_checker
        self.delta_max = neigh_delta_max

    def clear(self):
        self.verts[:self.cnt] = 0
        self.cnt = 0
        self.edges_parent.clear()
        self.edges_children.clear()
        self.cost_2_go.clear()

    def add_vert_to_tree(self, v, v_parent=None):
        self.verts[self.cnt] = v
        if v_parent is not None:
            self.edges_children[tuple(v_parent)].append(tuple(v))
            self.edges_parent[tuple(v)] = tuple(v_parent)
            self.cost_2_go[tuple(v)] = self.cost_2_go[tuple(v_parent)] + np.linalg.norm(v - v_parent)
        else:
            self.edges_parent[tuple(v)] = None
            self.cost_2_go[tuple(v)] = 0
        self.cnt += 1

    def check_transition(self, q_src, q_dst):
        transition = get_transition(q_src, q_dst, delta_min=1e-2)
        return self.coll_checker.are_states_collision_free(transition)

    def get_nearest_reachable_neigh(self, qt):
        dist = self.verts[:self.cnt] - qt
        i_min = np.argmin(np.hypot(dist[:, 0], dist[:, 1]))
        return self.verts[i_min]

    def rewire_connected(self, v):
        V, cost_2_go = self.verts[:self.cnt], self.cost_2_go
        within_ball = np.linalg.norm(V - v, axis=1) < self.delta_max
        V_neigh = V[within_ball]
        best_cost_through_neigh = cost_2_go[tuple(v)]
        costs_through_n = np.array([cost_2_go[tuple(v_n)] + np.linalg.norm(v_n - v) for v_n in V_neigh])
        indxs = np.argsort(costs_through_n)
        V_neigh = V_neigh[indxs]
        costs_through_n = costs_through_n[indxs]
        coll_free_mask = [self.check_transition(v, v_n) for v_n in V_neigh]
        for v_n_coll_free, cost_through_n in zip(V_neigh[coll_free_mask], costs_through_n[coll_free_mask]):
            if cost_through_n < best_cost_through_neigh:
                self.rewire_edge(v, v_n_coll_free, cost_through_n)
            break
        for v_n in V_neigh[coll_free_mask]:
            cost_through_v = cost_2_go[tuple(v)] + np.linalg.norm(v_n - v)
            if cost_through_v < cost_2_go[tuple(v_n)]:
                self.rewire_edge(v_n, v, cost_through_v)
                self.update_children_cost(tuple(v_n))

    def rewire_edge(self, v, v_parent_new, new_cost):
        self.cost_2_go[tuple(v)] = new_cost
        v_parent_old = self.edges_parent[tuple(v)]
        self.edges_parent[tuple(v)] = tuple(v_parent_new)
        self.edges_children[tuple(v_parent_new)].append(tuple(v))
        self.edges_children[v_parent_old].remove(tuple(v))

    def update_children_cost(self, v_p):
        for v_c in self.edges_children[v_p]:
            dist = np.linalg.norm(np.array(v_c) - np.array(v_p))
            new_cost = self.cost_2_go[v_p] + dist
            assert new_cost <= self.cost_2_go[v_c], f" {new_cost} {self.cost_2_go[v_c]}"
            self.cost_2_go[v_c] = self.cost_2_go[v_p] + dist
            self.update_children_cost(v_c)


class BiRRTST:
    """
    TODO: Could add a cost?
    TODO: Need to warm-start the goal-tree....
    TODO: Maybe actually faster to do some incremental rewiring???? Since we have a limited velocity change...
    """

    def __init__(
            self,
            q_lims,
            ts_obs,
            ts_p,
            ts_h,
            delta_max=.5,
            neigh_delta_max=0.5,
            speed_max=.1,
            tree_start_size_max=1_000,
            tree_goal_size_max=1_000,
            tree_warm_start=1_000,
            motion_model_poly_degree=3,
            coll_checker=None,
            naive_check=True
    ):
        t_e = 2
        self.scl = t_e / ts_h
        t_p = ts_p * self.scl
        speed_max_scl = speed_max / self.scl
        self.t_p = t_p
        self.t_e = t_e
        # TODO add scaling..
        self.coll_checker = coll_checker or CollisionChecker(
            scl=self.scl,
            ts_p=ts_p,
            ts_obs=ts_obs,
            motion_model_poly_degree=motion_model_poly_degree
        )
        self.delta_max = delta_max
        self.speed_max = speed_max_scl
        self.qt_lims_p = np.vstack([q_lims, [0, t_p]])
        self.qt_lims_s = np.vstack([q_lims, [t_p, t_e]])
        dim_q = self.qt_lims_p.shape[0] - 1
        dim_t = 1
        dim = dim_q + dim_t
        self.tree_s = TreeStart(size_max=tree_start_size_max, dim=dim, speed_max=speed_max_scl)
        self.tree_g = TreeGoal(size_max=tree_goal_size_max + tree_warm_start, dim=dim, speed_max=speed_max_scl)
        self.tree_g_ws = TreeStatic(
            size_max=tree_warm_start, dim=dim - 1, coll_checker=self.coll_checker, neigh_delta_max=neigh_delta_max
        )
        self.dim = dim
        self.naive_check = naive_check

    def reset(self):
        self.tree_s.clear()
        self.tree_g.clear()
        self.tree_g_ws.clear()
        self.coll_checker.clear()

    def interpolate_waypoints(self, path_qt_sparse):
        path_t = path_qt_sparse[:, -1] / self.scl
        path_pos = path_qt_sparse[:, :-1]
        ts = np.arange(0, path_t[-1] + 1)[::2]
        path = np.vstack([
            np.interp(ts, path_t, path_pos[:, 0]),
            np.interp(ts, path_t, path_pos[:, 1])
        ]).T
        path_qt = np.hstack(
            [path, ts.reshape(-1, 1)]
        )
        return path_qt

    def interpolate_static_path(self, path):
        path_t = np.linspace(self.t_p, self.t_e,
                             path.shape[0]) / self.scl  # TODO: need to think about time interpolation...
        t = np.arange(self.t_p / self.scl, self.t_e / self.scl + 1)
        path = np.vstack([
            np.interp(t, path_t, path[:, 0]),
            np.interp(t, path_t, path[:, 1])
        ]).T
        path_qt = np.hstack(
            [path, t.reshape(-1, 1) * self.scl]
        )
        return path_qt

    def warmstart_goal_tree(self, q_g):
        self.tree_g_ws.clear()
        self.tree_g_ws.add_vert_to_tree(q_g, v_parent=None)
        for i in range(self.tree_g_ws.verts.shape[0]):
            q = self.sample_static_coll_free()
            q_nearest = self.tree_g_ws.get_nearest_reachable_neigh(q)
            dist = np.linalg.norm(q - q_nearest)
            if dist > self.delta_max:
                beta = self.delta_max / dist
                q = (1 - beta) * q - beta * q_nearest
            if self.tree_g_ws.check_transition(q_nearest, q):
                self.tree_g_ws.add_vert_to_tree(q, v_parent=q_nearest)
                self.tree_g_ws.rewire_connected(q)

    def sample_static_coll_free(self):
        while True:
            qt = np.random.uniform(self.qt_lims_s[:-1, 0], self.qt_lims_s[:-1, 1])
            if self.coll_checker.is_state_collision_free(qt):
                break
        return qt

    def sample_within_goal_cone(self, q_goal):
        t_goal_upper = self.t_e
        speed_max = self.speed_max
        max_samples = 1_000
        qt = None
        cnt = 0
        while cnt < max_samples:
            cnt += 1
            qt = self.sample_static_coll_free_qt()
            q, t = qt[:-1], qt[-1]
            dq_goal = np.linalg.norm(q - q_goal)
            dt_goal = np.abs(t_goal_upper - t)
            if dt_goal == 0:
                continue
            s_goal = dq_goal / dt_goal
            if s_goal > speed_max or t > t_goal_upper:
                continue
            break
        if cnt >= max_samples:
            qt = None
        return qt

    def solve(
            self,
            q_s,
            q_g,
            nr_iters
    ):
        v_s = np.append(q_s, 0)
        v_g = np.append(q_g, self.t_p)
        path = np.array([]).reshape((0, self.dim))
        if not self.coll_checker.is_state_collision_free(v_s):
            info = {"iter": 0, "status": None}
            info["status"] = "infeasible"
            return info, path
        if self.naive_check:
            if self.check_transition(self.tree_s, v_s, v_g):
                info = {"iter": 1, "status": "naive_goal"}
                v_g = np.append(q_g, self.t_e)
                path = self.interpolate_waypoints(np.vstack([v_s, v_g]))
                return info, path
        # TODO: this should be done in a smarter way
        self.tree_s.clear()
        self.tree_g.clear()
        self.tree_s.add_vert_to_tree(v_s, v_parent=None)
        cnt = self.tree_g_ws.cnt
        for v in self.tree_g_ws.verts[:cnt]:
            self.tree_g.add_vert_to_tree(np.append(v, self.t_p), v_parent=None)
        _, qt_b_nearest = self.tree_g.get_nearest_reachable_neigh(v_s)
        if self.naive_check:
            if qt_b_nearest is not None and self.check_transition(self.tree_s, v_s, qt_b_nearest):
                info = {"iter": 1, "status": "naive_closest"}
                path_static_ws = self.tree_g_ws.get_path_from_vert(qt_b_nearest[:-1])
                path_ws = self.interpolate_static_path(path_static_ws)
                path_sparse = np.vstack([v_s, qt_b_nearest, path_ws[1:]])
                path = self.interpolate_waypoints(path_sparse)
                return info, path
        tree_a, tree_b = self.tree_s, self.tree_g
        cnt = 0
        for i in range(nr_iters):
            qt_a = self.grow_tree(tree_a, q_s)
            qt_b_nearest = None
            if qt_a is not None:
                _, qt_b_nearest = tree_b.get_nearest_reachable_neigh(qt_a)
            curr_is_start = tree_a == self.tree_s
            if curr_is_start:
                qt_src = qt_a
                qt_dst = qt_b_nearest
            else:
                qt_src = qt_b_nearest
                qt_dst = qt_a
            if qt_b_nearest is not None and self.check_transition(self.tree_s, qt_src, qt_dst):
                path_a = tree_a.get_path_from_vert(qt_a)
                path_b = tree_b.get_path_from_vert(qt_b_nearest)
                if curr_is_start:
                    path_static_ws = self.tree_g_ws.get_path_from_vert(path_b[-1, :-1])
                    path_ws = self.interpolate_static_path(path_static_ws)
                    path = np.vstack([path_a[::-1], path_b, path_ws[1:]])
                else:
                    path_static_ws = self.tree_g_ws.get_path_from_vert(path_a[-1, :-1])
                    path_ws = self.interpolate_static_path(path_static_ws)
                    path = np.vstack([path_b[::-1], path_a, path_ws[1:]])
                break
            if tree_a.cnt >= tree_a.cnt_max:
                break
            tree_a, tree_b = tree_b, tree_a
            cnt += 1
        info = {
            "iter": cnt,
            "status": "infeasible" if path.size == 0 else "success"
        }
        path = self.interpolate_waypoints(path) if path.size > 0 else path
        return info, path

    def grow_tree(self, tree, q_s):
        qt_free = self.sample_conditionally(q_s)
        if qt_free is None:
            raise RuntimeError("Cannot find a collision free configuration within cones...")
        delta_closest, qt_closest = tree.get_nearest_reachable_neigh(qt_free)
        if qt_closest is None:
            return None
        if delta_closest > self.delta_max:
            beta = self.delta_max / delta_closest
            qt_free = qt_closest * (1 - beta) + qt_free * beta
        if self.check_transition(tree, qt_closest, qt_free):
            tree.add_vert_to_tree(qt_free, qt_closest)
            return qt_free
        return None

    def sample_conditionally(self, q_start):
        t_start = 0
        t_goal_upper = self.t_e
        speed_max = self.speed_max
        max_samples = 1_000
        qt = None
        cnt = 0
        while cnt < max_samples:
            cnt += 1
            qt = self.sample_dynamic_coll_free_qt()
            q, t = qt[:-1], qt[-1]
            # check if within the state dependent cone...
            dq_start = np.linalg.norm(q - q_start)
            dt_start = np.abs(t_start - t)
            if dt_start == 0:
                continue
            s_start = dq_start / dt_start
            if s_start > speed_max or t < t_start:
                continue
            # dq_goal = np.linalg.norm(q - q_goal)
            # dt_goal = np.abs(t_goal_upper - t)
            # if dt_goal == 0:
            #     continue
            # s_goal = dq_goal / dt_goal
            # if s_goal > speed_max or t > t_goal_upper:
            #     continue
            break
        if cnt >= max_samples:
            qt = None
        return qt

    def sample_static_coll_free_qt(self):
        while True:
            qt = np.random.uniform(self.qt_lims_s[:, 0], self.qt_lims_s[:, 1])
            if self.coll_checker.is_state_collision_free(qt):
                break
        return qt

    def sample_dynamic_coll_free_qt(self):
        while True:
            qt = np.random.uniform(self.qt_lims_p[:, 0], self.qt_lims_p[:, 1])
            if self.coll_checker.is_state_collision_free(qt):
                break
        return qt

    def check_transition(self, tree, qt_src, qt_dst):
        transition = get_transition(qt_src, qt_dst, delta_min=1e-2)
        return self.coll_checker.are_states_collision_free(transition)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    import time

    q_lims = [-1, 1]
    t_e = 2
    box_w = .2
    box_h = .3
    box_x = 0
    box_t = .85
    box_dims = np.array([box_w, box_h])
    box_pos = np.array([box_x, box_t])
    box_pos_ll = box_pos - box_dims / 2

    nrgps = 32
    xs = np.linspace(*q_lims, nrgps)
    ts = np.linspace(0, t_e, nrgps)

    X, T = np.meshgrid(xs, ts, indexing="ij")
    D = np.ones((nrgps, nrgps))
    mask = (T >= box_t - box_h / 1.75) & (T <= box_t + box_h / 1.75)
    D[mask] = np.abs(X[mask] - box_pos[0]) - box_w / 2

    nr_gps = 32
    sdf_space = D
    grid_points = [np.linspace(*q_lims, nr_gps), np.linspace(0, t_e, nr_gps)]
    cc = CollisionChecker(grid_points, sdf_space)

    qt_s = np.array([-.0, 0.0])
    qt_e = np.array([0.0, 2.0])
    qts = np.vstack([qt_s, qt_e])

    planner = BiRRTST(
        q_lims,
        t_e,
        coll_checker=cc
    )

    q_s = qt_s[:-1]
    q_e = qt_e[:-1]

    time_s = time.time()
    path = planner.solve(
        q_s,
        q_e,
        nr_iters=10
    )
    print(time.time() - time_s)

    # fig, (ax1, ax) = plt.subplots(2, 1)
    fig, ax = plt.subplots()
    ax.add_patch(Rectangle(box_pos_ll, *box_dims))
    ax.contour(X, T, D, levels=(0.00, .05, 0.1))
    ax.scatter(qts[:, 0], qts[:, 1])

    s_max = planner.speed_max
    t_e = planner.t_e
    q_s = q_s[0]
    q_g = q_e[0]

    ax.add_patch(Polygon(np.vstack([
        [q_s, 0],
        [q_s + t_e * s_max, t_e],
        [q_s - t_e * s_max, t_e],
    ]), alpha=0.1))
    ax.add_patch(Polygon(np.vstack([
        [q_g, t_e],
        [q_g + t_e * s_max, 0],
        [q_g - t_e * s_max, 0],
    ]), alpha=0.1))

    for tree in (planner.tree_s, planner.tree_g):
        V = tree.get_vertices()
        ax.scatter(V[:, 0], V[:, 1], c="k")
        for v in V:
            v_p = tree.edges_parent[tuple(v)]
            if v_p is not None:
                line = np.vstack([v_p, v])
                ax.plot(line[:, 0], line[:, 1], color="k")

    ax.plot(path[:, 0], path[:, 1], color="r")
    ax.scatter(q_e, t_e)
    ax.set_xlim(*q_lims)
    ax.set_ylim(0, t_e)
    # candidates, candidates_cost = planner.get_candidates()
    # ax1.scatter(candidates[:, 0], candidates_cost)
    # ax1.set_xlim(*q_lims)
    plt.show()
