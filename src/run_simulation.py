import torch
import numpy as np
import os

from options import args_parser
from dataset import load_dataset
from results_saver import save_resuls
from scenario import create_dist_params
import models
from simulator import FLSimulator

from mylogger import logger_set_debug, logger


def run_simulation(args, path_project):
    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = "cuda" if args.gpu_id else "cpu"
    data_random_state = np.random.RandomState(seed=args.seed)
    p_list, user_silo_matrix, n_silos, n_users = create_dist_params(
        args.typical_scenaio, args.n_silos, args.n_users
    )

    # load data
    train_dataset, test_dataset, local_dataset_per_silos = load_dataset(
        data_random_state,
        args.dataset_name,
        path_project,
        args.n_users,
        args.n_silos,
        args.user_dist,
        args.silo_dist,
        args.user_alpha,
        args.silo_alpha,
        p_list,
        args.n_labels,
        user_silo_matrix,
        is_simulation=True,
        agg_strategy=args.agg_strategy,
    )

    # load model
    model = models.create_model(args.model_name, args.dataset_name, args.seed)

    # start training
    base_seed = np.random.RandomState(seed=args.seed).randint(2**32 - 1)

    if args.hp_tune:
        import optuna

        def objective(trial: optuna.Trial):
            # Target hyper parameters that wte want to optimize.
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 10)
            clipping_bound = trial.suggest_loguniform("clipping_bound", 1e-5, 10)
            # learning_rate = 0.02
            # clipping_bound = 0.86
            if args.agg_strategy == "ULDP-AVG":
                epochs = trial.suggest_int("epochs", 1, 10)
            else:
                epochs = args.epochs
            logger.info(
                "++++++++ Optuna setting: lr={}, clipping_bound={}, epochs={} ++++++++".format(
                    learning_rate,
                    clipping_bound,
                    epochs,
                )
            )
            error_rate_list = []
            for s in range(args.times):
                simulator = FLSimulator(
                    seed=s,
                    model=model,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    local_dataset_per_silos=local_dataset_per_silos,
                    n_silos=args.n_silos,
                    n_users=args.n_users,
                    device=device,
                    n_total_round=args.n_total_round,
                    n_silo_per_round=args.n_silo_per_round,
                    learning_rate=learning_rate,
                    local_batch_size=args.local_batch_size,
                    weight_decay=args.weight_decay,
                    client_optimizer=args.client_optimizer,
                    epochs=epochs,
                    agg_strategy=args.agg_strategy,
                    clipping_bound=clipping_bound,
                    sigma=args.sigma,
                    delta=args.delta,
                    group_k=args.group_k,
                )
                simulator.run()
                results = simulator.get_results()
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
            return error_rate

        study = optuna.create_study()
        study.optimize(objective, n_trials=100)

        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "trials": study.trials,
        }
    else:
        simulator = FLSimulator(
            seed=base_seed,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            local_dataset_per_silos=local_dataset_per_silos,
            n_silos=args.n_silos,
            n_users=args.n_users,
            device=device,
            n_total_round=args.n_total_round,
            n_silo_per_round=args.n_silo_per_round,
            learning_rate=args.learning_rate,
            local_batch_size=args.local_batch_size,
            weight_decay=args.weight_decay,
            client_optimizer=args.client_optimizer,
            epochs=args.epochs,
            agg_strategy=args.agg_strategy,
            clipping_bound=args.clipping_bound,
            sigma=args.sigma,
            delta=args.delta,
            group_k=args.group_k,
        )
        simulator.run()
        results = simulator.get_results()
    return results


if __name__ == "__main__":
    args = args_parser("simulation")
    src_path = os.path.dirname(os.path.abspath(__file__))
    path_project = os.path.dirname(src_path)
    if args.verbose:
        logger_set_debug()
    results = run_simulation(args, path_project)
    save_resuls(path_project, args, results)
