import torch
import numpy as np
import os

from options import args_parser
from dataset import load_dataset

# from results_saver import save_one_shot_results
import models
from server import FLServer
import ip_utils
from mylogger import logger_set_debug


if __name__ == "__main__":
    src_path = os.path.dirname(os.path.abspath(__file__))
    path_project = os.path.dirname(src_path)

    args = args_parser(path_project)

    if args.verbose:
        logger_set_debug()

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = "cuda" if args.gpu_id is not None else "cpu"
    data_random_state = np.random.RandomState(seed=args.seed)

    # load data
    train_dataset, test_dataset = load_dataset(
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
        is_simulation=False,
    )

    # load model
    model = models.create_model(args.model_name, args.dataset_name, args.seed)

    # start training
    silo_client_id_mapping = ip_utils.create_silo_client_id_mapping(args.n_silos)
    base_seed = np.random.RandomState(seed=args.seed).randint(2**32 - 1)
    server = FLServer(
        seed=base_seed,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_users=args.n_users,
        n_silos=args.n_silos,
        device=device,
        n_total_round=args.n_total_round,
        n_silo_per_round=args.n_silo_per_round,
        global_learning_rate=args.global_learning_rate,
        silo_client_id_mapping=silo_client_id_mapping,
        agg_strategy=args.agg_strategy,
        clipping_bound=args.clipping_bound,
        sigma=args.sigma,
        delta=args.delta,
        sampling_rate_q=args.sampling_rate_q,
        dataset_name=args.dataset_name,
        group_k=args.group_k,
        is_secure=args.secure_w,
    )
    server.run()

    # results = server.get_results()
    # save_one_shot_results(args, path_project, results, "server")
