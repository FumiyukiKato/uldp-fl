import os
import numpy as np
from personalized_utils import (
    make_epsilon_u,
    group_by_closest_below,
    show_specified_idx_result,
    run_with_specified_idx,
)
from options import args_parser
from dataset import HEART_DISEASE, TCGA_BRCA
from mylogger import logger_set_debug


"""
To show the comparison result between ULDP-AVG vs PULDP-AVG.
We can expect PULDP-AVG will be better than ULDP-AVG, because 
PULDP-AVG can utilize the data which is hold by liberal users who have larger privacy budgets.
"""

conservative_eps = 0.15
normal_eps = 1.0
liberal_eps = 5.0
epsilon_list = [conservative_eps, normal_eps, liberal_eps]
group_thresholds = epsilon_list
ratio_list = [0.34, 0.43, 0.23]

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
    q_step_size = args.q_step_size
    agg_strategy = "PULDP-AVG"

    random_state = np.random.RandomState(0)
    prefix_epsilon_u_list = []
    idx_per_group_list = []
    GOOD_IDX_SET = (1, 1, 1)

    for i in [GOOD_IDX_SET[0]]:
        for j in [GOOD_IDX_SET[1]]:
            for k in [GOOD_IDX_SET[2]]:
                idx_per_group = {conservative_eps: i, normal_eps: j, liberal_eps: k}
                idx_per_group_list.append(idx_per_group)
                print(idx_per_group)
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
                prefix_epsilon_u = list(epsilon_u.items())[:4]
                prefix_epsilon_u_list.append(prefix_epsilon_u)
                run_with_specified_idx(
                    epsilon_u,
                    sigma,
                    delta,
                    n_users,
                    n_round,
                    dataset_name,
                    q_step_size,
                    times,
                    idx_per_group=idx_per_group,
                    global_learning_rate=10.0,
                    local_learning_rate=0.001,
                    local_epochs=50,
                )
                show_specified_idx_result(
                    prefix_epsilon_u,
                    sigma,
                    delta,
                    n_users,
                    n_round,
                    dataset_name,
                    q_step_size,
                    idx_per_group,
                    validation_ratio=validation_ratio,
                    img_name=f"{dataset_name}-puldlpavg",
                    errorbar=False,
                )

    epsilon_list = [conservative_eps]
    group_thresholds = epsilon_list
    ratio_list = [1.0]
    BEST_IDX = 9

    for i in [BEST_IDX]:
        idx_per_group = {conservative_eps: i}
        idx_per_group_list.append(idx_per_group)
        print(idx_per_group)
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
        prefix_epsilon_u = list(epsilon_u.items())[:4]
        prefix_epsilon_u_list.append(prefix_epsilon_u)
        run_with_specified_idx(
            epsilon_u,
            sigma,
            delta,
            n_users,
            n_round,
            dataset_name,
            q_step_size,
            times,
            idx_per_group=idx_per_group,
            global_learning_rate=10.0,
            local_learning_rate=0.001,
            local_epochs=50,
        )
        show_specified_idx_result(
            prefix_epsilon_u,
            sigma,
            delta,
            n_users,
            n_round,
            dataset_name,
            q_step_size,
            idx_per_group,
            validation_ratio=validation_ratio,
            img_name=f"{dataset_name}-uldlpavg",
            errorbar=False,
        )

    show_specified_idx_result(
        prefix_epsilon_u_list,
        sigma,
        delta,
        n_users,
        n_round,
        dataset_name,
        q_step_size,
        idx_per_group_list,
        label_list=["PULDP-AVG", "ULDP-AVG"],
        validation_ratio=validation_ratio,
        img_name=f"{dataset_name}-comparison",
        errorbar=False,
    )
