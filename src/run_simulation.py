import torch
import numpy as np
import os

from options import args_parser
from dataset import load_dataset
from results_saver import save_resuls
from scenario import create_dist_params
import models
from simulator import FLSimulator

from mylogger import logger_set_debug

if __name__ == "__main__":
    args = args_parser("simulation")

    if args.verbose:
        logger_set_debug()

    path_project = os.path.abspath("..")
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
    )

    # load model
    model = models.create_model(args.model_name, args.dataset_name, args.seed)

    # start training
    base_seed = np.random.RandomState(seed=args.seed).randint(2**32 - 1)
    simulator = FLSimulator(
        seed=base_seed,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        local_dataset_per_silos=local_dataset_per_silos,
        n_silos=args.n_silos,
        device=device,
        n_total_round=args.n_total_round,
        n_silo_per_round=args.n_silo_per_round,
        lr=args.lr,
        local_batch_size=args.local_batch_size,
        weight_decay=args.weight_decay,
        client_optimizer=args.client_optimizer,
        epochs=args.epochs,
        agg_strategy=args.agg_strategy,
        clipping_bound=args.clipping_bound,
        sigma=args.sigma,
        delta=args.delta,
    )
    simulator.run()

    results = simulator.get_results()
    save_resuls(path_project, args, results)
