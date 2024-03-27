import numpy as np

from mpenv.obstacles import compute_rot_matrices


def compute_global_XY_points(nr_grid_points=32):
    grid_x, grid_y = np.linspace(-1, 1, nr_grid_points), np.linspace(-1, 1, nr_grid_points)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y, indexing='ij')
    XY = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T
    return XY


def compute_sdf_for_rect(points_local, dim_box):
    b = np.array(dim_box) / 2
    p = np.abs(points_local)
    d = np.linalg.norm(np.maximum(p - b, np.zeros(2, )), axis=1) - np.min(np.maximum(b - p, np.zeros(2, )), axis=1)
    return d


class SDFEstimator:

    def __init__(
            self,
            T_obs,
            T_pred,
            N=3,
            grid_points=32,
            distance_margin=0.1
    ):
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.N = N
        self.W = 2
        self.N_g = grid_points
        ts_observer = np.arange(T_obs)
        self.X = np.vander(ts_observer, N=N)  # [T_obs, N]
        self.X_T_X = self.X.T @ self.X  # [N, N]
        ts_pred = np.arange(T_obs - 1, T_obs - 1 + T_pred)
        self.X_pred = np.vander(ts_pred, N=N)
        self.X_pred_d = np.vander(ts_pred, N=N - 1)
        self.set_global_grid_points_and_mesh(nr_grid_points=self.N_g)
        self.ts = np.arange(self.T_pred)
        self.grid_points = [self.xs, self.ys, self.ts]
        self.Y_obs, self.V_pred, self.Y_pred = None, None, None
        self.distance_margin = distance_margin
        self.SDF_obs = None

    def set_global_grid_points_and_mesh(self, nr_grid_points=32):
        self.xs, self.ys = np.linspace(-1, 1, nr_grid_points), np.linspace(-1, 1, nr_grid_points)
        grid_X, grid_Y = np.meshgrid(self.xs, self.ys, indexing='ij')
        self.XY_g = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T

    def estimate_obstacle_motion(self, observation):
        """
        observation: [p_observed_0 ... p_observed_N_obs], [dim_0, ..., dim_N_obs]
        where
            p_observed_i: [T_obs, W]
        """
        p_observations, dims_obs = observation
        N_obs = len(dims_obs)
        self.Y_obs = np.hstack(p_observations)
        # X.T @ X  [N, N]
        # X  [T_obs, N]
        # Y  [T_obs, W]
        B = np.linalg.solve(self.X_T_X, self.X.T @ self.Y_obs)
        self.Y_pred = self.X_pred @ B
        B_d = B[:-1]
        B_d[0, :] *= 2
        self.V_pred = self.X_pred_d @ B_d
        self.Y_pred = self.Y_pred.reshape((self.T_pred, N_obs, self.W))
        self.V_pred = self.V_pred.reshape((self.T_pred, N_obs, self.W))
        # TODO: this could be vectorized
        ds = []
        for i, dims_i in enumerate(dims_obs):
            d_i = []
            Rs = compute_rot_matrices(np.hstack([self.V_pred[:, i, :], np.zeros((self.V_pred.shape[0], 1))]))
            for j in range(self.T_pred):
                p = self.Y_pred[j, i, :]
                R = Rs[:, :, j].T
                XY_l = R[:2, :2].T @ (self.XY_g - p).T
                d_i_t = compute_sdf_for_rect(XY_l.T, dims_i[:2])
                d_i.append(d_i_t)
            ds.append(np.stack(d_i))
        SDF_all_obs = np.stack(ds).reshape((-1, self.T_pred, self.N_g, self.N_g))
        SDF_all_obs = np.transpose(SDF_all_obs, (0, 2, 3, 1))
        # D = D_all_obs.min(axis=0)
        return SDF_all_obs


class MotionObstacleEstimator:

    def __init__(
            self,
            T_obs,
            N=3
    ):
        self.T_obs = T_obs
        self.N = N
        self.W = 2
        ts_observer = np.arange(T_obs)
        self.X = np.vander(ts_observer, N=N)  # [T_obs, N]
        self.X_T_X = self.X.T @ self.X  # [N, N]

    def estimate_obstacle_motion(self, observation):
        """
        observation: [p_observed_0 ... p_observed_N_obs], [radius_0, ..., radius_N_obs]
        where
            p_observed_i: [T_obs, W]
        """
        p_observations, radiuses = observation
        Y_obs = np.hstack(p_observations)
        # X.T @ X  [N, N]
        # X  [T_obs, N]
        # Y  [T_obs, W]
        Th = np.linalg.solve(self.X_T_X, self.X.T @ Y_obs)
        return Th, np.array(radiuses).ravel()
