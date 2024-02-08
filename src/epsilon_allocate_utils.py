from typing import Dict, List
import numpy as np
import noise_utils


def make_epsilon_u(
    epsilon=1.0,
    n_users=0,
    dist="homo",
    epsilon_list=[],
    ratio_list=[],
    random_state: np.random.RandomState = None,
):
    if dist == "homo":
        epsilon_u = {user_id: epsilon for user_id in range(n_users)}
    elif dist == "hetero":
        assert len(epsilon_list) > 0 and len(ratio_list) > 0
        epsilon_u_list = random_state.choice(epsilon_list, size=n_users, p=ratio_list)
        epsilon_u = {user_id: epsilon_u_list[user_id] for user_id in range(n_users)}
    else:
        raise ValueError(f"invalid dist {dist}")
    return epsilon_u


def make_static_params(
    epsilon_u_dct,
    delta,
    sigma,
    n_total_round,
    idx_per_group,
    q_step_size=None,
    static_q_u_list=None,
):
    C_u_dct = {}
    q_u_dct = {}

    if static_q_u_list is not None:
        q_u_list = static_q_u_list
    else:
        # exponential
        MAX_Q_LIST_SIZE = 30
        n_of_q_u = MAX_Q_LIST_SIZE
        q_u_list = []
        init_q_u = 1.0
        for _ in range(n_of_q_u):
            q_u_list.append(init_q_u)
            init_q_u *= q_step_size

    C_and_q_per_group = {}
    for group_eps, idx in idx_per_group.items():
        q_u = q_u_list[idx]
        C_u, _eps, _ = noise_utils.from_q_u(
            q_u=q_u, delta=delta, epsilon_u=group_eps, sigma=sigma, T=n_total_round
        )
        assert _eps <= group_eps, f"_eps={_eps} > eps_u={group_eps}"
        C_and_q_per_group[group_eps] = (C_u, q_u)

    for user_id, eps_u in epsilon_u_dct.items():
        C_u, q_u = C_and_q_per_group[eps_u]
        C_u_dct[user_id] = C_u
        q_u_dct[user_id] = q_u

    return C_u_dct, q_u_dct


def group_by_closest_below(epsilon_u_dct: Dict, group_thresholds: List):
    minimum = min(epsilon_u_dct.values())
    group_thresholds = set(group_thresholds) | {minimum}
    grouped = {
        g: [] for g in group_thresholds
    }  # Initialize the dictionary with empty lists for each group threshold
    for key, value in epsilon_u_dct.items():
        # Find the closest group threshold that is less than or equal to the value
        closest_group = max([g for g in group_thresholds if g <= value], default=None)
        # If a suitable group is found, append the key to the corresponding list
        if closest_group is not None:
            grouped[closest_group].append(key)

    return grouped
