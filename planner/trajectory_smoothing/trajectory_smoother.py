import time

import cvxpy as cp
import numpy as np
import scipy as sp


class TrajectorySmootherWithDistance:

    def __init__(
            self,
            u_max,
            ts_p,
            dim=2,
            distance_margin_dynamic=-0.2,
            distance_margin_static=-0.05,
            alpha_goal_range=(-2, 0),
            alpha_distance_dyn_range=(2, -1),
            beta_distance_dyn_range=5,
            beta_distance_sta_range=5,
            alpha_distance_sta_range=(2, -1)
    ):
        W = 2
        self.W = W
        I_2 = np.eye(W)
        O_2 = np.zeros((W, W))
        self.A = np.block(
            [
                [I_2, I_2],
                [O_2, I_2]]
        )
        self.B = np.block([
            [I_2 * .5],
            [I_2]
        ])
        self.dim = dim
        self.u_max = u_max
        self.ts_p = ts_p
        self.distance_margin_dynamic = distance_margin_dynamic
        self.distance_margin_static = distance_margin_static
        self.alpha_goal_range = alpha_goal_range
        self.alpha_distance_dyn_range = alpha_distance_dyn_range
        self.beta_distance_dyn_range = beta_distance_dyn_range
        self.beta_distance_sta_range = beta_distance_sta_range
        self.alpha_distance_sta_range = alpha_distance_sta_range

    def vectorize_hyper_planes(self, bps, bns):
        nr_obst = bps.shape[1]
        time_steps = bps.shape[0]
        indxs = np.array([
            (i * nr_obst + j, i * 2 + k) for i in range(time_steps) for j in range(nr_obst) for k in range(2)
        ])
        As = sp.sparse.coo_matrix(
            (bns.reshape(-1), (indxs[:, 0], indxs[:, 1])),
            shape=(time_steps * nr_obst, time_steps * 2)
        )
        Bs = np.sum(bns * bps, axis=-1).reshape(-1)
        return As, Bs

    def compute_distances(self, ps, A, B, nr_obstacles):
        # time_steps must be uneven
        time_steps = ps.shape[0]
        indxs_o = np.arange(0, time_steps, 2)
        indxs_m = np.arange(1, time_steps, 2)
        ps_o = ps[indxs_o]
        ps_m = ps[indxs_m]
        D_o_f = -A @ cp.vec(ps_o, order="C") + B
        D_o = cp.reshape(D_o_f, (indxs_o.size, nr_obstacles), order="C")
        A_ = A.tocsr()
        D_1 = -A_[:-nr_obstacles, :-2] @ cp.vec(ps_m, order="C") + B[:-nr_obstacles]
        D_2 = -A_[nr_obstacles:, 2:] @ cp.vec(ps_m, order="C") + B[nr_obstacles:]
        D_m = cp.reshape(cp.hstack([D_1, D_2]), (indxs_m.size, 2 * nr_obstacles), order="C")
        D_m_f = cp.reshape(D_m, indxs_m.size * 2 * nr_obstacles, order="C")
        return (D_o_f, D_o), (D_m_f, D_m)

    def compute_distance_cost(self, D_o, D_m, alphas, distance_margin):
        alphas_o = alphas[::2]
        alphas_m = alphas[1:-1:2]
        return (
                alphas_o @ cp.maximum(cp.max(D_o, axis=1), distance_margin)
                +
                alphas_m @ cp.maximum(cp.max(D_m, axis=1), distance_margin)
        )

    def solve(
            self,
            x_s,
            x_g,
            path,
            obstacles_p0_static,
            obstacles_p0_grad_static,
            obstacles_p0_dyn,
            obstacles_p0_grad_dyn,
            slacks_penalty=None
    ):
        """
        obstacles_p0_static: [O, T, 2]
        """
        time_s = time.time()
        bps_d, bns_d = obstacles_p0_dyn, obstacles_p0_grad_dyn
        bps_s, bns_s = obstacles_p0_static, obstacles_p0_grad_static

        m = path.shape[0]
        n = 2
        M = 2 * m - 1
        P = self.ts_p

        ps = cp.Variable((M, n))
        vs = cp.Variable((M, n))
        xs = cp.hstack([ps, vs])
        us = cp.Variable((M - 1, n))
        N_static_obst = obstacles_p0_static.shape[1]
        As, Bs = self.vectorize_hyper_planes(bps_s, bns_s)
        (D_o_f_s, D_o_s), (D_m_f_s, D_m_s) = self.compute_distances(
            ps, As, Bs, nr_obstacles=N_static_obst
        )
        N_dyn_obst = obstacles_p0_dyn.shape[1]
        dists_dyn = 0
        if N_dyn_obst:
            Ad, Bd = self.vectorize_hyper_planes(bps_d, bns_d)
            (D_o_f_d, D_o_d), (D_m_f_d, D_m_d) = self.compute_distances(
                ps[:P], Ad, Bd, nr_obstacles=N_dyn_obst
            )
            alphas = np.exp(np.linspace(*self.alpha_distance_dyn_range, self.ts_p)) * self.beta_distance_dyn_range
            dists_dyn += self.compute_distance_cost(D_o_d, D_m_d, alphas, self.distance_margin_dynamic)
        dists_sta = 0
        if N_static_obst:
            alphas = np.exp(np.linspace(*self.alpha_distance_sta_range, M)) * self.beta_distance_sta_range
            dists_sta += self.compute_distance_cost(D_o_s, D_m_s, alphas, self.distance_margin_static)
        if N_dyn_obst:
            distance_vector = cp.hstack([
                D_o_f_s, D_m_f_s, D_o_f_d, D_m_f_d
            ])
        else:
            distance_vector = cp.hstack([
                D_o_f_s, D_m_f_s
            ])
        alphas = np.exp(np.linspace(*self.alpha_goal_range, M)).reshape((-1, 1))
        loss = (
                dists_dyn + dists_sta
                +
                cp.sum(cp.norm(us, 1))
                +
                alphas.T @ cp.norm(xs[:, :2] - x_g[:2].reshape((1, -1)), axis=1)
        )
        constraints = (
                [
                    xs[0] == x_s.ravel(),
                    xs[-1] == x_g.ravel(),
                    cp.norm(us, axis=1) <= self.u_max,
                    xs[:, :n] <= 1,
                    -xs[:, :n] <= 1
                ] + [
                    xs[i + 1, :] == self.A @ xs[i] + self.B @ us[i] for i in range(M - 1)
                ]
        )
        if slacks_penalty is None:
            constraints += [distance_vector <= 0]
        else:
            slacks = cp.Variable(distance_vector.size)
            loss += -cp.sum(slacks) * slacks_penalty
            constraints += [
                distance_vector <= -slacks,
                slacks <= 0
            ]
        problem = cp.Problem(cp.Minimize(loss), constraints)
        loss = problem.solve(solver="ECOS")
        time_e = time.time() - time_s
        info = {"status": problem.status, "t": time_e, "loss": loss}
        if slacks_penalty is not None:
            info["slack_min"] = slacks.value.min() if slacks.value is not None else None
        return info, xs.value, us.value
