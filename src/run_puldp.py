import numpy as np

import os

path_project = os.path.dirname(os.path.abspath("."))
import sys

sys.path.append(os.path.join(path_project, "src"))
sys.path.append(os.path.join(path_project, "exp/script"))

img_path = os.path.join(path_project, "exp", "img")
pickle_path = os.path.join(path_project, "exp", "pickle")
results_path = os.path.join(path_project, "exp", "results")

from personalized_utils import (
    make_q_c_curve,
    plot_q_c_curve,
    make_epsilon_u,
    group_by_closest_below,
    static_optimization,
    show_static_optimization_result,
    prepare_grid_search,
    run_online_optimization,
    show_online_optimization_result,
    run_with_specified_idx,
    show_specified_idx_result,
)
from mylogger import logger_set_warning, logger_set_info, logger_set_debug


if __name__ == "__main__":
    # WITHOUT
    sigma = 1.0
    epsilon_list = [5.0]
    group_thresholds = [5.0]
    ratio_list = [1.0]
    delta = 1e-5
    n_round = 50
    dataset_name = "mnist"
    q_step_size = 0.7
    times = 2
    validation_ratio = 0.0
    gpu_id = None
    global_learning_rate = 5.0
    local_learning_rate = 0.001
    local_epochs = 50

    for n_users in [100]:
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
        opt_strategy = prepare_grid_search(epsilon_u, start_idx=0, end_idx=1)

        logger_set_info()
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
            user_dist="uniform-iid",
            silo_dist="uniform",
            gpu_id=gpu_id,
            parallelized=True,
            force_update=True,
        )
        min_idx, min_loss = show_static_optimization_result(
            epsilon_u,
            sigma,
            delta,
            n_users,
            n_round,
            dataset_name,
            q_step_size,
            n_silos=4,
            opt_strategy=opt_strategy,
            validation_ratio=validation_ratio,
            train_loss=False,
            user_dist="uniform-iid",
            img_name=f"{dataset_name}-uniform-users-{n_users}",
            global_learning_rate=global_learning_rate,
            local_learning_rate=local_learning_rate,
            local_epochs=local_epochs,
        )
        print(min_idx, min_loss)

    # OFFLINE HPO
    # sigma = 1.0
    # n_users = 400
    # epsilon_list = [0.15, 3.0, 5.0]
    # group_thresholds = epsilon_list
    # ratio_list = [0.6, 0.25, 0.15]
    # delta = 1e-5
    # n_round = 20
    # dataset_name = 'heart_disease'
    # q_step_size = 0.7
    # times = 10
    # validation_ratio = 0.0
    # static_q_u_list = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]

    # random_state = np.random.RandomState(0)
    # epsilon_u = make_epsilon_u(n_users=n_users, dist="hetero", epsilon_list=epsilon_list, ratio_list=ratio_list, random_state=random_state)
    # grouped = group_by_closest_below(epsilon_u_dct=epsilon_u, group_thresholds=group_thresholds)
    # epsilon_u = {}
    # for eps_u, user_ids in grouped.items():
    #     for user_id in user_ids:
    #         epsilon_u[user_id] = eps_u
    # opt_strategy = prepare_grid_search(epsilon_u, start_idx=0, end_idx=7)

    # logger_set_info()
    # static_optimization(
    #     epsilon_u, sigma, delta, n_users, n_round, dataset_name, times,
    #     q_step_size, opt_strategy=opt_strategy, global_learning_rate=10.0,
    #     local_learning_rate=0.001, local_epochs=30, validation_ratio=validation_ratio,
    #     user_dist="uniform-iid", silo_dist="uniform",
    #     static_q_u_list=static_q_u_list, global_learning_rate=10.0, local_learning_rate=0.001, local_epochs=30, force_update=True,
    # )
    # min_idx, min_loss = show_static_optimization_result(
    #     epsilon_u, sigma, delta, n_users, n_round, dataset_name,
    #     q_step_size, n_silos=4, opt_strategy=opt_strategy, validation_ratio=validation_ratio,
    #     train_loss=False, img_name=f"heart_disease-users-{n_users}", is_3d=True,
    #     static_q_u_list=static_q_u_list, user_dist="uniform-iid", global_learning_rate=10.0, local_learning_rate=0.001, local_epochs=30
    # )
    # print(f"min_idx = {min_idx}, min_loss = {min_loss}")

    # ONLINE HPO
    # QCTest
    # n_users = 400
    # sigma = 1.0
    # epsilon_list = [0.15, 3.0, 5.0]
    # group_thresholds = epsilon_list
    # ratio_list = [0.6, 0.25, 0.15]
    # delta = 1e-5
    # n_round = 30
    # dataset_name = 'heart_disease'
    # q_step_size = 0.8
    # initial_q_u_list = [0.1, 0.5, 1.0]
    # validation_ratio = 0.5
    # times = 10
    # agg_strategy = "PULDP-AVG-QCTest"
    # with_momentum = True
    # step_decay = True
    # hp_baseline = None

    # logger_set_info()

    # for initial_q_u in initial_q_u_list:
    #     random_state = np.random.RandomState(0)
    #     epsilon_u_dct = make_epsilon_u(n_users=n_users, dist="hetero", epsilon_list=epsilon_list, ratio_list=ratio_list, random_state=random_state)
    #     grouped = group_by_closest_below(epsilon_u_dct=epsilon_u_dct, group_thresholds=group_thresholds)
    #     epsilon_u = {}
    #     for eps_u, user_ids in grouped.items():
    #         for user_id in user_ids:
    #             epsilon_u[user_id] = eps_u
    #     run_online_optimization(
    #         epsilon_u, sigma, delta, n_users, n_round, dataset_name, q_step_size,
    #         group_thresholds, times, global_learning_rate=10.0, local_learning_rate=0.001,
    #         local_epochs=30, validation_ratio=validation_ratio, agg_strategy=agg_strategy,
    #         with_momentum=with_momentum, step_decay=step_decay, hp_baseline=hp_baseline,
    #         user_dist="uniform-iid", silo_dist="uniform", initial_q_u=initial_q_u, global_learning_rate=10.0, local_learning_rate=0.001, local_epochs=30,
    #     )
    #     x, acc_means, acc_stds = show_online_optimization_result(
    #         epsilon_u, sigma, delta, n_users, n_round, dataset_name, q_step_size,
    #         validation_ratio=validation_ratio, agg_strategy=agg_strategy, with_momentum=with_momentum,  initial_q_u=initial_q_u,
    #         step_decay=step_decay, hp_baseline=hp_baseline, errorbar=False, img_name=f"{dataset_name}-{agg_strategy}-{initial_q_u}",
    #         global_learning_rate=10.0, local_learning_rate=0.001, local_epochs=30,
    #     )

    # QCTest
    # n_users = 400
    # sigma = 1.0
    # epsilon_list = [0.15, 3.0, 5.0]
    # group_thresholds = epsilon_list
    # ratio_list = [0.6, 0.25, 0.15]
    # delta = 1e-5
    # n_round = 30
    # dataset_name = 'heart_disease'
    # q_step_size = 0.8
    # initial_q_u_list = [0.1, 0.5, 1.0]
    # validation_ratio = 0.5
    # times = 10
    # agg_strategy = "PULDP-AVG-QCTrain"
    # with_momentum = True
    # step_decay = True
    # hp_baseline = None
    # momentum_weight = 0.5

    # logger_set_info()

    # for initial_q_u in initial_q_u_list:
    #     random_state = np.random.RandomState(0)
    #     epsilon_u_dct = make_epsilon_u(n_users=n_users, dist="hetero", epsilon_list=epsilon_list, ratio_list=ratio_list, random_state=random_state)
    #     grouped = group_by_closest_below(epsilon_u_dct=epsilon_u_dct, group_thresholds=group_thresholds)
    #     epsilon_u = {}
    #     for eps_u, user_ids in grouped.items():
    #         for user_id in user_ids:
    #             epsilon_u[user_id] = eps_u
    #     if not is_from_pickle:
    #         run_online_optimization(
    #             epsilon_u, sigma, delta, n_users, n_round, dataset_name, q_step_size,
    #             group_thresholds, times, global_learning_rate=10.0, local_learning_rate=0.001,
    #             local_epochs=30, validation_ratio=validation_ratio, agg_strategy=agg_strategy,
    #             with_momentum=with_momentum, step_decay=step_decay, hp_baseline=hp_baseline,
    #             user_dist="uniform-iid", silo_dist="uniform", initial_q_u=initial_q_u, global_learning_rate=10.0, local_learning_rate=0.001, local_epochs=30,
    #         )
    #     x, acc_means, acc_stds = show_online_optimization_result(
    #         epsilon_u, sigma, delta, n_users, n_round, dataset_name, q_step_size,
    #         validation_ratio=validation_ratio, agg_strategy=agg_strategy, with_momentum=with_momentum, initial_q_u=initial_q_u,
    #         step_decay=step_decay, hp_baseline=hp_baseline, errorbar=False, img_name=f"{dataset_name}-{agg_strategy}-{initial_q_u}-{step_decay}",
    #         global_learning_rate=10.0, local_learning_rate=0.001, local_epochs=30
    #     )
