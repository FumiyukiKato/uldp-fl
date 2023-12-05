import copy
import torch
import numpy as np
import os

from options import args_parser
from dataset import HEART_DISEASE, TCGA_BRCA, load_dataset

from results_saver import save_one_shot_results, args_to_hash
import models
from secure_simulator import SecureWeightingFLSimulator

from simulator import FLSimulator, TrainNanError

from mylogger import logger_set_debug


def run_simulation(args, path_project, data_seed=None):
    if args.dry_run:
        # print(str(args))
        print("========> Hash value: ", args_to_hash(args))
        exit(0)
    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = "cuda" if args.gpu_id is not None else "cpu"
    if data_seed is None:
        data_seed = args.seed
    data_random_state = np.random.RandomState(seed=data_seed)

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
        args.n_labels,
        is_simulation=True,
    )

    # load model
    model = models.create_model(args.model_name, args.dataset_name, args.seed)

    # start training
    base_seed = np.random.RandomState(seed=args.seed).randint(2**32 - 1)
    if args.secure_w:
        simulator = SecureWeightingFLSimulator(
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
            local_learning_rate=args.local_learning_rate,
            global_learning_rate=args.global_learning_rate,
            local_batch_size=args.local_batch_size,
            weight_decay=args.weight_decay,
            client_optimizer=args.client_optimizer,
            local_epochs=args.local_epochs,
            agg_strategy=args.agg_strategy,
            clipping_bound=args.clipping_bound,
            sigma=args.sigma,
            delta=args.delta,
            group_k=args.group_k,
            dataset_name=args.dataset_name,
            sampling_rate_q=args.sampling_rate_q,
        )
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
            local_learning_rate=args.local_learning_rate,
            global_learning_rate=args.global_learning_rate,
            local_batch_size=args.local_batch_size,
            weight_decay=args.weight_decay,
            client_optimizer=args.client_optimizer,
            local_epochs=args.local_epochs,
            agg_strategy=args.agg_strategy,
            clipping_bound=args.clipping_bound,
            sigma=args.sigma,
            delta=args.delta,
            group_k=args.group_k,
            dataset_name=args.dataset_name,
            sampling_rate_q=args.sampling_rate_q,
            C_u=args.C_u,
            q_u=args.q_u,
            epsilon_u=args.epsilon_u,
            group_thresholds=args.group_thresholds,
            q_step_size=args.q_step_size,
            validation_ratio=args.validation_ratio,
        )
    simulator.run()
    results = simulator.get_results()
    return results


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

    results_list = []
    org_args = copy.deepcopy(args)
    for i in range(args.times):
        print("======== TIME:", i, "start")
        args.seed = args.seed + i
        try:
            sim_results = run_simulation(args, path_project)
            results_list.append(sim_results)
        except TrainNanError:
            results_list.append("LOSS IS NAN")
        except AssertionError:
            results_list.append("Assertion Error")

    save_one_shot_results(org_args, path_project, {"exp": results_list}, "sim")
