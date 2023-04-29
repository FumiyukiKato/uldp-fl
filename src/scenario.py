import numpy as np


def create_dist_params(typical_scenaio: int, n_users: int, n_silos: int):
    """
    Create special distribution parameters for specific scenarios.
    """
    p_list, user_silo_matrix = None, None
    if typical_scenaio is None:
        pass
    elif typical_scenaio == 1:
        n_users = 1000
        n_silos = 4
        user_silo_matrix_1 = [[0.5, 0.2, 0.2, 0.1] for u in range(500)]
        user_silo_matrix_2 = [[0.1, 0.5, 0.3, 0.1] for u in range(500)]
        user_silo_matrix = np.array(user_silo_matrix_1 + user_silo_matrix_2)
    elif typical_scenaio == 2:
        n_silos = 4
        p_list = [0.5, 0.25, 0.2, 0.05]
    else:
        pass
    return p_list, user_silo_matrix, n_silos, n_users
