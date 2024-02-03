import os
import numpy as np
from personalized_utils import (
    make_epsilon_u,
    group_by_closest_below,
    prepare_grid_search,
    static_optimization,
    show_static_optimization_result,
)
from options import args_parser
from dataset import HEART_DISEASE, TCGA_BRCA
from mylogger import logger_set_debug


"""
To show the static optimization result,
describing the evaluation results with various q, C
"""

if __name__ == "__main__":
    src_path = os.path.dirname(os.path.abspath(__file__))
    path_project = os.path.dirname(src_path)
    args = args_parser(path_project)
    if args.verbose:
        logger_set_debug()

    if args.dataset_name == HEART_DISEASE:
        from flamby_utils.heart_disease import update_args

        args = update_args(args)

    elif args.dataset_name == TCGA_BRCA:
        from flamby_utils.tcga_brca import update_args

        args = update_args(args)

    sigma = args.sigma
    epsilon_list = args.epsilon_list
    group_thresholds = args.group_thresholds
    ratio_list = args.ratio_list
    delta = args.delta
    n_round = args.n_total_round
    dataset_name = args.dataset_name
    q_step_size = args.q_step_size
    times = args.times
    validation_ratio = args.validation_ratio
    n_users = args.n_users
    global_learning_rate = args.global_learning_rate
    local_learning_rate = args.local_learning_rate
    local_epochs = args.local_epochs

    random_state = np.random.RandomState(0)
    epsilon_u = make_epsilon_u(
        n_users=n_users,
        dist="hetero",
        epsilon_list=epsilon_list,
        ratio_list=ratio_list,
        random_state=random_state,
    )
    grouped = group_by_closest_below(
        epsilon_u_dct=epsilon_u, group_thresholds=group_thresholds
    )
    epsilon_u = {}
    for eps_u, user_ids in grouped.items():
        for user_id in user_ids:
            epsilon_u[user_id] = eps_u
    opt_strategy = prepare_grid_search(epsilon_u, start_idx=0, end_idx=12)

    static_optimization(
        epsilon_u,
        sigma,
        delta,
        n_users,
        n_round,
        dataset_name,
        times,
        q_step_size,
        opt_strategy=opt_strategy,
        global_learning_rate=global_learning_rate,
        local_learning_rate=local_learning_rate,
        local_epochs=local_epochs,
        validation_ratio=validation_ratio,
    )
    min_idx, min_loss = show_static_optimization_result(
        epsilon_u,
        sigma,
        delta,
        n_users,
        n_round,
        dataset_name,
        q_step_size,
        n_silos=args.n_silos,
        opt_strategy=opt_strategy,
        validation_ratio=validation_ratio,
        train_loss=True,
        img_name=f"{dataset_name}-users-{n_users}",
    )
    print(min_idx, min_loss)
