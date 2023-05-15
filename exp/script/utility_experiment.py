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
from results_saver import load_best_params, save_resuls, save_best_params, args_to_hash
from run_simulation import run_simulation
from mylogger import logger_set_debug, logger


def build_exp_paramerters(default_args, dataset, dist, method, n_users):
    copy_args = copy.deepcopy(default_args)

    copy_args.dataset_name = dataset
    if dataset in ["mnist", "cifar10", "cifar100"]:
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
    elif dataset == "heart_disease":
        from flamby_utils.heart_disease import update_args

        copy_args = update_args(copy_args)
    elif dataset == "isic":
        from flamby_utils.isic import update_args

        copy_args = update_args(copy_args)
    elif dataset == "tcga_brca":
        from flamby_utils.tcga_brca import update_args

        copy_args = update_args(copy_args)
    else:
        raise NotImplementedError

    copy_args.n_users = n_users

    if method == "DEFAULT":
        pass
    elif method == "ULDP-GROUP":
        copy_args.sigma = 5.0
        copy_args.clipping_bound = 1.0
        copy_args.delta = 0.00001
    elif method in ["ULDP-SGD", "ULDP-SGD-w"]:
        copy_args.sigma = 5.0
        copy_args.clipping_bound = 1.0
        copy_args.delta = 0.00001
    elif method in ["ULDP-AVG", "ULDP-AVG-w"]:
        copy_args.sigma = 5.0
        copy_args.clipping_bound = 1.0
        copy_args.delta = 0.00001

    return copy_args


def hyper_parameter_tuning(args, path_project):
    import optuna

    N_SEED = 1  # Baysian searching like `optuna` is relatively robust for seed
    N_TRIALS = 100

    original_args = copy.deepcopy(args)
    result_details = []

    hash_args = args_to_hash(args)

    def objective(trial: optuna.Trial):
        # Target hyper parameters that wte want to optimize.
        hyper_params = {}

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 100, log=True)
        args.learning_rate = learning_rate
        logger.debug(
            "++++++++ Optuna setting: learning_rate={} ++++++++".format(learning_rate)
        )
        hyper_params["learning_rate"] = learning_rate

        clipping_bound = trial.suggest_float("clipping_bound", 1e-4, 100, log=True)
        args.clipping_bound = clipping_bound
        logger.debug(
            "++++++++ Optuna setting: clipping_bound={} ++++++++".format(clipping_bound)
        )
        hyper_params["clipping_bound"] = clipping_bound

        if args.agg_strategy in ["ULDP-AVG", "ULDP-GROUP", "DEFAULT", "ULDP-NAIVE"]:
            local_epochs = trial.suggest_int("local_epochs", 1, 40, step=2)
            args.local_epochs = local_epochs
            logger.debug(
                "++++++++ Optuna setting: local_epochs={} ++++++++".format(local_epochs)
            )
            hyper_params["local_epochs"] = local_epochs

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
        logger.debug(
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
        study_name=f"{hash_args}",
        storage=f"sqlite:///{hash_args}-storage.db",
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=original_args.seed),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)
    try:
        best_params = study.best_params
        best_value = study.best_value
    except ValueError as e:
        logger.warning(f"No trials are completed yet. {str(e)}")
        best_params = "Too Bad."
        best_value = 0.0

    return {
        "best_params": best_params,
        "best_value": best_value,
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
        hp_results = hyper_parameter_tuning(args, path_project)
        save_best_params(args, path_project, hp_results["best_params"])
        exp_results = {"hp_results": hp_results}
    else:
        best_params = load_best_params(args, path_project)
        if type(best_params) is dict:
            args.learning_rate = best_params["learning_rate"]
            args.clipping_bound = best_params["clipping_bound"]
            if args.agg_strategy in ["ULDP-AVG", "ULDP-GROUP", "DEFAULT", "ULDP-NAIVE"]:
                args.local_epochs = best_params["local_epochs"]
            logger.info(
                "++++++++ Load Best params: learning_rate={}, clipping_bound={}, local_epochs={} ++++++++".format(
                    args.learning_rate, args.clipping_bound, args.local_epochs
                )
            )
        elif type(best_params) is str:
            logger.warning(
                "++++++++ All params are too Bad and use default params ++++++++"
            )
        else:
            logger.warning("++++++++ Best params Not Found ++++++++")
            raise ValueError("Best params Not Found")
        results_list = []
        for i in range(args.times):
            args.seed = args.seed + i
            try:
                sim_results = run_simulation(args, path_project)
                results_list.append(sim_results["global"])
            except OverflowError as e:
                logger.error(f"OverflowError: {str(e)}")
                results_list.append("LOSS IS NAN")
        exp_results = results_list

    save_resuls(
        args,
        path_project,
        {"exp": exp_results},
        hp=original_args.hyper_parameter_tuning,
    )
