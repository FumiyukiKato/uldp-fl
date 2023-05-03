import torch
import numpy as np
import os

from options import args_parser
from dataset import load_dataset

# from results_saver import save_resuls
from scenario import create_dist_params
import models
from silo import FLSilo
from mylogger import logger_set_debug

if __name__ == "__main__":
    args = args_parser("silo")

    if args.verbose:
        logger_set_debug()

    path_project = os.path.abspath("..")
    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = "cuda" if args.gpu_id else "cpu"
    data_random_state = np.random.RandomState(seed=args.seed)
    assert (
        args.silo_id < args.n_silos
    ), "silo_id should be less than n_silos, and start from 0"
    p_list, user_silo_matrix, n_silos, n_users = create_dist_params(
        args.typical_scenaio, args.n_silos, args.n_users
    )

    # load data
    train_dataset, test_dataset, user_hist, user_ids = load_dataset(
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
        silo_id=args.silo_id,
    )

    # load model
    model = models.create_model(args.model_name, args.dataset_name, args.seed)

    # start training
    client_id = args.silo_id + 1
    base_seed = np.random.RandomState(seed=args.seed).randint(2**32 - 1)
    silo = FLSilo(
        seed=base_seed,
        model=model,
        agg_strategy=args.agg_strategy,
        local_train_dataset=train_dataset,
        local_test_dataset=test_dataset,
        user_histogram=user_hist,
        user_ids_of_local_train_dataset=user_ids,
        learning_rate=args.learning_rate,
        local_batch_size=args.local_batch_size,
        client_optimizer=args.client_optimizer,
        epochs=args.epochs,
        device=device,
        silo_id=args.silo_id,
        client_id=client_id,
        n_total_round=args.n_total_round,
        weight_decay=args.weight_decay,
        local_sigma=args.sigma,
        local_delta=args.delta,
        local_clipping_bound=args.clipping_bound,
        group_k=args.group_k,
    )
    silo.run()

    # results = silo.get_results()
    # save_resuls(path_project, args, results)
