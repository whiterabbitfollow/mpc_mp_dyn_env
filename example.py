import pathlib
import pickle
import copy
import time
from collections import Counter
import pprint

import numpy as np

from mpenv.environment import MPEnv
from planner.mpc import MPCPlanner


PATH_WS = pathlib.Path("ws")
PATH_WS.mkdir(exist_ok=True)


def run_example(seed=1, env_nr=0, time_steps_max=200):
    v_max = 0.2
    T_horizon = 50
    u_max = 0.01
    T_obs = 5
    T_pred = 9
    alpha_distance_range = (1, -1)
    distance_margin_dyn = -0.2
    planner = MPCPlanner(
        u_max=u_max,
        T_obs=T_obs,
        T_pred=T_pred,
        T_horizon=T_horizon,
        v_max=v_max,
        naive_check=True,
        kwargs_traj=dict(
            alpha_distance_dyn_range=alpha_distance_range,
            alpha_distance_sta_range=alpha_distance_range,
            distance_margin_dynamic=distance_margin_dyn
        )
    )
    env = MPEnv(kwargs_obstacles=dict(
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
    env.reset(seed=seed, hard=True)
    print("warm starting planner")
    run_warm_start(env, planner, env_nr, seed)
    print("running scenario")
    return run_scenario(env, planner, env_nr, seed, time_steps_max=time_steps_max)


def run_warm_start(env, planner, env_nr, seed):
    if (PATH_WS / f"env_{env_nr}_seed_{seed}.pckl").exists():
        return True
    np.random.seed(seed)
    x, p_g = env.reset(seed=seed)
    planner.reset(static_obstacles=env.world.obstacles.static_obstacles)
    planner.warm_start(p_g)
    tree_g_ws = planner.global_planner.tree_g_ws
    warm_start_data = (
        (x, p_g),
        tree_g_ws.verts.copy(),
        copy.deepcopy(dict(tree_g_ws.cost_2_go)),
        copy.deepcopy(dict(tree_g_ws.edges_children)),
        copy.deepcopy(dict(tree_g_ws.edges_parent)),
        tree_g_ws.cnt
    )
    try:
        _, path = planner.global_planner.solve(
            q_s=x[:2], q_g=p_g, nr_iters=2000
        )
    except Exception as e:
        return None
    if path.size > 0:
        with (PATH_WS / f"env_{env_nr}_seed_{seed}.pckl").open("wb") as fp:
            pickle.dump(warm_start_data, fp)
    return path.size != 0


def run_scenario(env, planner, env_nr, seed, time_steps_max=100):
    path_file = (PATH_WS / f"env_{env_nr}_seed_{seed}.pckl")
    if not path_file.is_file():
        return None
    with path_file.open("rb") as fp:
        warm_start_data = pickle.load(fp)
    np.random.seed(seed)
    x, p_g = env.reset(seed=seed)
    planner.reset(static_obstacles=env.world.obstacles.static_obstacles)
    tree_g_ws = planner.global_planner.tree_g_ws
    (
        (x_c, p_g_c),
        tree_g_ws.verts,
        tree_g_ws.cost_2_go,
        tree_g_ws.edges_children,
        tree_g_ws.edges_parent,
        tree_g_ws.cnt
    ) = warm_start_data
    assert (x == x_c).all()
    collision_free = True
    no_action_produced = False
    times = []
    cnt = 0
    cnt_max = time_steps_max
    cnt_in_goal = 0
    error = False
    info = None
    collision_dynamic = False
    collision_static = False
    planner_stats = Counter()
    planner_stats["gp_iter_min"] = np.inf
    planner_stats["gp_t_min"] = np.inf
    planner_stats["to_t_min"] = np.inf
    while collision_free and cnt < cnt_max:
        t = env.t
        dims = []
        positions = []
        for o in env.world.obstacles.dynamic_obstacles:
            lim = o.positions.shape[0]
            indexes = list(map(lambda x: x % lim, range(t + 1 - planner.T_obs, t + 1)))
            positions.append(o.positions[indexes, :])
            dims.append(o.dimensions)
        obs_repr = (positions, dims)
        state = (env.x.copy(), p_g, obs_repr)
        try:
            time_s = time.time()
            info, actions = planner.get_action(state)
            time_e = time.time() - time_s
        except Exception as e:
            error = True
            break
        no_action_produced = actions is None
        if no_action_produced:
            break
        _, collision_free, is_done = env.step(actions)
        env.render(pause_time=0.1)
        cnt_in_goal += np.linalg.norm(env.x[:2].ravel() - p_g) < 0.1
        times.append(time_e)
        if info.get("gp") and info["gp"]["status"] != "infeasible":
            gp_iter = info["gp"]["iter"]
            gp_t = info["gp"]["t"]
            planner_stats["gp_cnt"] += 1
            planner_stats["gp_iter_acc"] += gp_iter
            planner_stats["gp_iter_max"] = max(planner_stats["gp_iter_max"], gp_iter)
            planner_stats["gp_iter_min"] = min(planner_stats["gp_iter_min"], gp_iter)
            planner_stats["gp_t_acc"] += gp_t
            planner_stats["gp_t_max"] = max(planner_stats["gp_t_max"], gp_t)
            planner_stats["gp_t_min"] = min(planner_stats["gp_t_min"], gp_t)
        if info.get("to") and info["to"]["status"] != "infeasible":
            to_t = info["to"]["t_complete"]
            planner_stats["to_cnt"] += 1
            planner_stats["to_t_acc"] += to_t
            planner_stats["to_t_max"] = max(planner_stats["to_t_max"], to_t)
            planner_stats["to_t_min"] = min(planner_stats["to_t_min"], to_t)
            if info["to"].get("slack_max"):
                slack = info["to"].get("slack_max")
                planner_stats["to_slack_max_cnt"] += 1
                planner_stats["to_slack_max_acc"] += slack
        cnt += 1
    if not collision_free:
        collision, names = env.world.robot.collision_manager.in_collision_other(
            env.world.obstacles.collision_manager, return_names=True
        )
        collision_dynamic = False
        collision_static = False
        for _, obst_name in names:
            collision_dynamic = collision_dynamic or obst_name.startswith("obstacle_dynamic")
            collision_static = collision_static or obst_name.startswith("obstacle_static")
    done = cnt >= cnt_max and collision_free and not no_action_produced
    gp_infeasible = False
    to_infeasible = False
    if not done:
        if info.get("gp"):
            gp_infeasible = info["gp"]["status"] == "infeasible"
        if info.get("to"):
            to_infeasible = info["to"]["status"] == "infeasible"
    stats = {
        **{
            "done": done,
            "no_action_produced": no_action_produced,
            "collision": not collision_free,
            "collision_dynamic": collision_dynamic,
            "collision_static": collision_static,
            "time_eval_mean": np.array(times).mean() if times else 0,
            "cnt_in_goal": int(cnt_in_goal),
            "goal_rate": cnt_in_goal / cnt if cnt > 0 else 0,
            "gp_infeasible": gp_infeasible,
            "to_infeasible": to_infeasible,
            "error": error,
            "cnt": cnt
        },
        **planner_stats
    }
    return stats


if __name__ == "__main__":
    stats = run_example()
    pprint.pp(stats)