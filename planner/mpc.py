import time

import numpy as np

from planner.motion_estimation.vanilla_ls_est import MotionObstacleEstimator
from planner.global_planners.bi_rrt_st import BiRRTST, get_transition
from planner.trajectory_smoothing.trajectory_smoother import TrajectorySmootherWithDistance


class MPCPlanner:

    def __init__(
            self,
            T_obs=8,
            T_pred=10,
            T_horizon=30,
            v_max=.1,
            u_max=.025,
            kwargs_traj=None,
            poly_degree=3,
            delta_max=1,
            naive_check=True
    ):
        self.T_pred = T_pred
        self.T_horizon = T_horizon
        self.T_obs = T_obs
        self.motion_est = MotionObstacleEstimator(
            T_obs,
            N=poly_degree
        )
        self.global_planner = BiRRTST(
            q_lims=[
                [-1, 1],
                [-1, 1]
            ],
            ts_obs=self.T_obs,
            ts_p=self.T_pred,
            ts_h=self.T_horizon,
            delta_max=delta_max,
            speed_max=v_max,
            tree_start_size_max=1_000,
            tree_goal_size_max=1_000,
            tree_warm_start=1_000,
            motion_model_poly_degree=poly_degree,
            naive_check=naive_check
        )
        kwargs_traj = kwargs_traj or {}
        self.traj_opt = TrajectorySmootherWithDistance(
            u_max=u_max,
            ts_p=T_pred,
            **kwargs_traj
        )
        self.path = None
        self.path_raw = None
        self.path_smooth = None
        self.actions = None
        self.path_qt = None
        self.obs_p0_dyn = None
        self.obs_p0_grad_dyn = None
        self.obs_p0_static = None
        self.obs_p0_grad_static = None
        self.path_qt_rrt = None

    def reset(
            self,
            static_obstacles
    ):
        self.path = None
        self.path_smooth = None
        self.actions = None
        self.global_planner.reset()
        self.global_planner.coll_checker.clear()
        self.global_planner.coll_checker.add_static_obstacles(static_obstacles)

    def warm_start(self, p_g):
        self.global_planner.warmstart_goal_tree(q_g=p_g)

    def is_path_collision_free(self, xs):
        scl = self.global_planner.coll_checker.scl
        xs[:, -1] *= scl
        status = False
        for x_src, x_dst in zip(xs[:-1], xs[1:]):
            line = get_transition(x_src, x_dst)
            status = self.global_planner.coll_checker.are_states_collision_free(line)
            if not status:
                break
        xs[:, -1] /= scl
        return status

    def are_path_states_collision_free(self, xs):
        scl = self.global_planner.coll_checker.scl
        xs[:, -1] *= scl
        status = self.global_planner.coll_checker.are_states_collision_free(xs)
        xs[:, -1] /= scl
        return status

    def get_action(self, state):
        """
        TODO: reuse path from previous iteration
        sdf_all_obst: [N_o, T, X, Y]
        """
        x, q_g, obs_repr = state
        p_g = q_g[:2]
        x_g = np.hstack([q_g, np.zeros(p_g.size, )]).reshape((-1, 1))
        q = x[:2, 0]
        dynamic_obstacles_detected = len(obs_repr) > 0
        if dynamic_obstacles_detected:
            Th, rs = self.motion_est.estimate_obstacle_motion(obs_repr)
            self.global_planner.coll_checker.Th = Th
            self.global_planner.coll_checker.rs = rs
        else:
            self.global_planner.coll_checker.clear_dynamic_obstacles()
        info = {"mpc": ""}
        if self.path is not None:
            self.path = self.path[1::2]
            self.path_raw = self.path_raw[1:]
            path_qt = np.hstack([
                self.path_raw,
                np.arange(0, self.T_horizon).reshape((-1, 1))
            ]
            )
            if self.are_path_states_collision_free(path_qt):
                self.path = np.vstack([self.path, self.path[-1, :]])
                self.path_qt = np.hstack([
                    self.path, np.arange(0, self.T_horizon+1).reshape((-1, 1))[::2]
                ]
                )
                info["mpc"] = "path reused"
            else:
                self.path = None
                self.path_qt = None
        if self.path is None:
            time_s = time.time()
            info_gp, path_qt_rrt = self.global_planner.solve(
                q_s=q,
                q_g=p_g,
                nr_iters=2_000
            )
            time_e = time.time() - time_s
            info["gp"] = {**info_gp, **{"t": time_e}}
            self.path_qt_rrt = path_qt_rrt
            self.path_qt = path_qt_rrt
            if self.path_qt.size == 0:
                return info, None
            self.path = self.path_qt[:, :-1]
        else:
            self.path_qt_rrt = None
        self.obs_p0_dyn, self.obs_p0_grad_dyn = [], []
        self.obs_p0_static, self.obs_p0_grad_static = self.global_planner.coll_checker.get_static_obstacles_closest_points_and_normals(
            self.path
        )
        self.obs_p0_dyn, self.obs_p0_grad_dyn = self.global_planner.coll_checker.get_dynamic_obstacles_closest_points_and_normals(
            self.path_qt
        )
        time_s = time.time()
        info_to_ = {}
        for slack_penalty in (None, 100, 50, 10, 1):
            try:
                info_to, xs, us = self.traj_opt.solve(
                    x_s=x,
                    x_g=x_g,
                    path=self.path,
                    obstacles_p0_static=self.obs_p0_static,
                    obstacles_p0_grad_static=self.obs_p0_grad_static,
                    obstacles_p0_dyn=self.obs_p0_dyn,
                    obstacles_p0_grad_dyn=self.obs_p0_grad_dyn,
                    slacks_penalty=slack_penalty
                )
            except:
                info_to_[f"slack_{slack_penalty}"] = "Error"
                info_to = {}
                xs, us = None, None
                continue
            if xs is None:
                info_to_[f"slack_{slack_penalty}"] = "Infeasible"
            else:
                info_to_[f"slack_{slack_penalty}"] = "Feasible"
            if xs is not None:
                break
        time_e = time.time() - time_s
        info["to"] = {**info_to, **info_to_, **{"t_complete": time_e}}
        if xs is not None and us is not None:
            W = 2
            self.path_raw = xs[:, :W]
            self.path = xs[:, :W]
            self.us = us
            return info, us[0, :]
        else:
            return info, None
