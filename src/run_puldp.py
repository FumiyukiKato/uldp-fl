import os

path_project = os.path.dirname(os.path.abspath("."))
import sys

sys.path.append(os.path.join(path_project, "src"))
sys.path.append(os.path.join(path_project, "exp/script"))
img_path = os.path.join(path_project, "exp", "img")
pickle_path = os.path.join(path_project, "exp", "pickle")
results_path = os.path.join(path_project, "exp", "results")

from personalized_utils import (
    init_mnist_param,
    run_online_optimization,
    show_online_optimization_result,
    prepare_grid_search,
)
from mylogger import logger_set_warning, logger_set_info, logger_set_debug


if __name__ == "__main__":
    # random baselines for MNIST n=500
    # logger_set_info()
    # fed_sim_params = init_mnist_param(
    #     parallelized=False, n_users=500, times=5)
    # fed_sim_params.epsilon_list = [0.15, 3.0, 5.0]
    # fed_sim_params.group_thresholds = [0.15, 3.0, 5.0]
    # fed_sim_params.ratio_list = [0.6, 0.25, 0.15]
    # fed_sim_params.validation_ratio = 0.5
    # fed_sim_params.agg_strategy = "PULDP-AVG-QCTrain"
    # fed_sim_params.hp_baseline = "random"

    # run_online_optimization(fed_sim_params, force_update=False)
    # x, acc_means, acc_stds = show_online_optimization_result(
    #     fed_sim_params, errorbar=False,
    #     img_name=f"{fed_sim_params.dataset_name}-{fed_sim_params.hp_baseline}"
    # )

    # fed_sim_params = init_mnist_param(
    #     parallelized=False, n_users=500, times=5)
    # fed_sim_params.epsilon_list = [0.15, 3.0, 5.0]
    # fed_sim_params.group_thresholds = [0.15, 3.0, 5.0]
    # fed_sim_params.ratio_list = [0.6, 0.25, 0.15]
    # fed_sim_params.validation_ratio = 0.5
    # fed_sim_params.agg_strategy = "PULDP-AVG-QCTrain"
    # fed_sim_params.hp_baseline = "random-log"

    # run_online_optimization(fed_sim_params, force_update=False)
    # x, acc_means, acc_stds = show_online_optimization_result(
    #     fed_sim_params, errorbar=False,
    #     img_name=f"{fed_sim_params.dataset_name}-{fed_sim_params.hp_baseline}"
    # )

    fed_sim_params = init_mnist_param(parallelized=False, n_users=500, times=5)
    fed_sim_params.epsilon_list = [0.15, 3.0, 5.0]
    fed_sim_params.group_thresholds = [0.15, 3.0, 5.0]
    fed_sim_params.ratio_list = [0.6, 0.25, 0.15]
    fed_sim_params.validation_ratio = 0.5
    fed_sim_params.agg_strategy = "PULDP-AVG-QCTrain"
    fed_sim_params.hp_baseline = "random-updown"

    initial_q_u_list = [0.1, 0.5, 1.0]
    for initial_q_u in initial_q_u_list:
        fed_sim_params.initial_q_u = initial_q_u
        run_online_optimization(fed_sim_params, force_update=False)
        x, acc_means, acc_stds = show_online_optimization_result(
            fed_sim_params,
            errorbar=False,
            img_name=f"{fed_sim_params.dataset_name}-{initial_q_u}-{fed_sim_params.hp_baseline}",
        )

    # random baselines for MNIST n=2000
    # logger_set_info()
    # fed_sim_params = init_mnist_param(
    #     parallelized=True, n_users=2000, times=3, gpu_id=0)
    # fed_sim_params.epsilon_list = [0.15, 3.0, 5.0]
    # fed_sim_params.group_thresholds = [0.15, 3.0, 5.0]
    # fed_sim_params.ratio_list = [0.6, 0.25, 0.15]
    # fed_sim_params.validation_ratio = 0.5
    # fed_sim_params.agg_strategy = "PULDP-AVG-QCTrain"
    # fed_sim_params.hp_baseline = "random"

    # run_online_optimization(fed_sim_params, force_update=False)
    # x, acc_means, acc_stds = show_online_optimization_result(
    #     fed_sim_params, errorbar=False,
    #     img_name=f"{fed_sim_params.dataset_name}-{fed_sim_params.hp_baseline}-users-{2000}"
    # )

    # fed_sim_params = init_mnist_param(
    #     parallelized=True, n_users=2000, times=3, gpu_id=0)
    # fed_sim_params.epsilon_list = [0.15, 3.0, 5.0]
    # fed_sim_params.group_thresholds = [0.15, 3.0, 5.0]
    # fed_sim_params.ratio_list = [0.6, 0.25, 0.15]
    # fed_sim_params.validation_ratio = 0.5
    # fed_sim_params.agg_strategy = "PULDP-AVG-QCTrain"
    # fed_sim_params.hp_baseline = "random-log"

    # run_online_optimization(fed_sim_params, force_update=False)
    # x, acc_means, acc_stds = show_online_optimization_result(
    #     fed_sim_params, errorbar=False,
    #     img_name=f"{fed_sim_params.dataset_name}-{fed_sim_params.hp_baseline}-users-{2000}"
    # )

    # fed_sim_params = init_mnist_param(parallelized=True, n_users=2000, times=3, gpu_id=0)
    # fed_sim_params.epsilon_list = [0.15, 3.0, 5.0]
    # fed_sim_params.group_thresholds = [0.15, 3.0, 5.0]
    # fed_sim_params.ratio_list = [0.6, 0.25, 0.15]
    # fed_sim_params.validation_ratio = 0.5
    # fed_sim_params.agg_strategy = "PULDP-AVG-QCTrain"
    # fed_sim_params.hp_baseline = "random-updown"

    # initial_q_u_list = [0.1, 0.5, 1.0]
    # for initial_q_u in initial_q_u_list:
    #     fed_sim_params.initial_q_u = initial_q_u
    #     run_online_optimization(fed_sim_params, force_update=False)
    #     x, acc_means, acc_stds = show_online_optimization_result(
    #         fed_sim_params,
    #         errorbar=False,
    #         img_name=f"{fed_sim_params.dataset_name}-{initial_q_u}-{fed_sim_params.hp_baseline}-users-{2000}",
    #     )
