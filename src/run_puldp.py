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
    run_with_specified_idx,
    show_specified_idx_result,
)
from mylogger import logger_set_warning, logger_set_info, logger_set_debug


if __name__ == "__main__":
    logger_set_info()
    fed_sim_params = init_mnist_param(
        n_users=500, eps_u=0.15, parallelized=True, gpu_id=3
    )
    fed_sim_params.epsilon_list = [0.15]
    fed_sim_params.group_thresholds = [0.15]
    fed_sim_params.ratio_list = [1.0]
    fed_sim_params.validation_ratio = 0.5
    fed_sim_params.times = 5
    fed_sim_params.agg_strategy = "PULDP-AVG"
    static_q_u_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
    best_idx_per_group = {0.15: 4}

    run_with_specified_idx(
        fed_sim_params,
        best_idx_per_group,
        static_q_u_list=static_q_u_list,
        force_update=False,
    )
    x, acc_means, acc_stds = show_specified_idx_result(
        fed_sim_params,
        best_idx_per_group,
        static_q_u_list=static_q_u_list,
        errorbar=False,
        img_name=f"{fed_sim_params.dataset_name}-ULDPAVG-users-{500}",
    )
