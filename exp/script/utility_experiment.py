# flake8: noqa E402
import os
import sys
import numpy as np
import copy

exp_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_project = os.path.dirname(exp_path)
src_path = os.path.join(path_project, "src")
sys.path.append(src_path)

from options import args_parser
from results_saver import load_best_params, save_resuls, save_best_params
from run_simulation import run_simulation
from mylogger import logger_set_debug, logger


def build_exp_paramerters(default_args, dataset, dist, method, n_users):
    copy_args = copy.deepcopy(default_args)

    copy_args.n_total_round = 5

    if dataset == "mnist":
        copy_args.dataset_name = dataset
    else:
        raise NotImplementedError

    if dist == 0:
        copy_args.user_dist = "uniform-iid"
        copy_args.silo_dist = "uniform"
    elif dist == 1:
        copy_args.user_dist = "zipf-noniid"
        copy_args.user_alpha = 0.3
        copy_args.silo_dist = "zipf"
        copy_args.silo_alpha = 1.5
        copy_args.n_labels = 1
    else:
        raise NotImplementedError

    copy_args.n_users = n_users

    if method == "DEFAULT":
        pass
    elif method == "ULDP-GROUP":
        copy_args.group_k = 2
        copy_args.sigma = 5.0
        copy_args.clipping_bound = 1.0
        copy_args.delta = 0.00001
    elif method == "ULDP-SGD":
        copy_args.sigma = 5.0
        copy_args.clipping_bound = 1.0
        copy_args.delta = 0.00001
    elif method == "ULDP-AVG":
        copy_args.sigma = 5.0
        copy_args.clipping_bound = 1.0
        copy_args.delta = 0.00001

    return copy_args


def hyper_parameter_tuning(args):
    import optuna

    N_SEED = 2  # Baysian searching like `optuna` is relatively robust for seed
    N_TRIALS = 100

    original_args = copy.deepcopy(args)
    result_details = []

    def objective(trial: optuna.Trial):
        # Target hyper parameters that wte want to optimize.
        hyper_params = {}

        learning_rate = trial.suggest_float("learning_rate", 1e-3, 100, log=True)
        args.learning_rate = learning_rate
        logger.info(
            "++++++++ Optuna setting: learning_rate={} ++++++++".format(learning_rate)
        )
        hyper_params["learning_rate"] = learning_rate

        clipping_bound = trial.suggest_float("clipping_bound", 1e-3, 100, log=True)
        args.clipping_bound = clipping_bound
        logger.info(
            "++++++++ Optuna setting: clipping_bound={} ++++++++".format(clipping_bound)
        )
        hyper_params["clipping_bound"] = clipping_bound

        if args.agg_strategy in ["ULDP-AVG", "ULDP-GROUP", "DEFAULT"]:
            epochs = trial.suggest_int("epochs", 1, 30, step=5)
            args.epochs = epochs
            logger.info("++++++++ Optuna setting: epochs={} ++++++++".format(epochs))
            hyper_params["epochs"] = epochs

        error_rate_list = []
        for i in range(N_SEED):
            args.seed = original_args.seed + i
            results = run_simulation(
                args,
                path_project,
                trial,
                # Using only trainig data in hyper parameter search
                data_seed=original_args.seed,
            )
            test_acc = results["global"]["global_test"][-1][1]
            error_rate = 1 - test_acc
            error_rate_list.append(error_rate)
        error_rate = np.mean(error_rate_list)
        error_std = np.std(error_rate_list)
        logger.info(
            "++++++ Optuna result: error_rate={}, error_std={} ++++++".format(
                error_rate, error_std
            )
        )
        result_details.append(
            {
                "params": hyper_params,
                "error_rate": error_rate,
                "error_std": error_std,
            }
        )
        return error_rate

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    save_best_params(original_args, path_project, study.best_params)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "result_details": result_details,
    }


if __name__ == "__main__":
    original_args = args_parser(path_project)
    args = build_exp_paramerters(
        original_args,
        original_args.dataset_name,
        original_args.exp_dist,
        original_args.agg_strategy,
        original_args.n_users,
    )
    if args.verbose:
        logger_set_debug()

    if args.hyper_parameter_tuning:
        hp_results = hyper_parameter_tuning(args)
        results = {"hp_results": hp_results}
    else:
        best_params = load_best_params(args, path_project)
        if best_params:
            args.learning_rate = best_params["learning_rate"]
            args.clipping_bound = best_params["clipping_bound"]
            if args.agg_strategy in ["ULDP-AVG", "ULDP-GROUP", "DEFAULT"]:
                args.epochs = best_params["epochs"]
            logger.info(
                "++++++++ Using Best params: learning_rate={}, clipping_bound={}, epochs={} ++++++++".format(
                    args.learning_rate, args.clipping_bound, args.epochs
                )
            )
        else:
            logger.warning("++++++++ Best params Not Found ++++++++")
        results_list = []
        for i in range(args.times):
            args.seed = args.seed + i
            sim_results = run_simulation(args, path_project)
            results_list.append(sim_results["global"])

    save_resuls(original_args, path_project, {"exp": results_list})
