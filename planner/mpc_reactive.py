import time

from planner.global_planners.bi_rrt_st import BiRRTST, get_transition

import numpy as np


class StaticObstacleEstimator:

    def __init__(
            self,
    ):
        self.T_obs = 1
        self.N = 1
        self.W = 2
        ts_observer = np.arange(self.T_obs)
        self.X = np.vander(ts_observer, N=self.N)        # [T_obs, N]
        self.X_T_X = self.X.T @ self.X                   # [N, N]

    def estimate_obstacle_motion(self, observation):
        """
        observation: [p_observed_0 ... p_observed_N_obs], [radius_0, ..., radius_N_obs]
        where
            p_observed_i: [T_obs, W]
        """
        p_observations, radiuses = observation
        Y_obs = np.hstack(p_observations)
        Y_obs = Y_obs[:1, :]
        # X.T @ X  [N, N]
        # X  [T_obs, N]
        # Y  [T_obs, W]
        Th = np.linalg.solve(self.X_T_X, self.X.T @ Y_obs)
        return Th, np.array(radiuses).ravel()


class ReactiveMPCPlanner:

    def __init__(
            self,
            T_pred=10,
            T_horizon=30,
            v_max=.1,
            u_max=.025,
            kwargs_traj=None
    ):
        self.T_pred = T_pred
        self.T_horizon = T_horizon
        self.T_obs = 1
        self.motion_est = StaticObstacleEstimator()
        self.global_planner = BiRRTST(
            q_lims=[
                [-1, 1],
                [-1, 1]
            ],
            ts_obs=self.T_obs,
            ts_p=self.T_pred,
            ts_h=self.T_horizon,
            delta_max=0.5,
            speed_max=v_max * 100,
            tree_start_size_max=1_000,
            tree_goal_size_max=1_000,
            tree_warm_start=7_00,
            motion_model_poly_degree=1
        )
        self.traj_opt = TrajectorySmootherWithDistanceNew(
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
            # self.path_raw = self.path_raw[2:]
            self.path_raw = self.path_raw[1:]
            assert np.isclose(self.path[0], q).all(), f"{self.path[0]} vs {q}"
            # path_qt = np.hstack([
            #     self.path_raw,
            #     np.arange(0, self.T_horizon-.5, .5).reshape((-1, 1))
            # ]
            # )
            path_qt = np.hstack([
                self.path_raw,
                np.arange(0, self.T_horizon).reshape((-1, 1))
            ]
            )
            if self.is_path_collision_free(path_qt):
                # TODO: missing the complete smooth path to check that...
                self.path = np.vstack([self.path, self.path[-1, :]])
                self.path_qt = np.hstack([
                    self.path, np.arange(0, self.T_horizon+1).reshape((-1, 1))[::2]
                ]
                )
                # self.path_qt[-1, -1] += 1 # Should not be needed
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
        # p = self.T_pred
        # self.obs_p0_dyn, self.obs_p0_grad_dyn = self.get_dyn_obstacles_closest_points_and_gradients(
        #     self.path_qt[:p], N_obst_dyn
        # )
        # if not self.path_qt.shape[0] == 31:
        #     print("")
        #     assert self.path_qt.shape[0] == 31, f"info: {info} {self.path_qt.shape[0]}"
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
            self.path = xs[:, :W] # [::2]
            # self.path = xs[:, :W]
            # path_qt = np.hstack([
            #     self.path_raw,
            #     np.arange(0, self.T_horizon+1).reshape((-1, 1))
            # ])
            # assert self.is_path_collision_free(path_qt), "Collision"
            self.us = us
            return info, us[0, :]
            # return info, us[:2, :]
        else:
            return info, None
