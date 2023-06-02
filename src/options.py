import os
import argparse
import yaml

from mylogger import logger


DEFAULT_CONFIG_PATH = "src/default_params.yaml"


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def pretty_print_args(args: argparse.Namespace):
    parameter_info_str = "\n Parameters:"
    for key, value in vars(args).items():
        parameter_info_str += "\n"
        parameter_info_str += f"\t{key}: {value}"
    logger.info(parameter_info_str)


def build_default_args(path_project):
    config = load_config(os.path.join(path_project, DEFAULT_CONFIG_PATH))
    return argparse.Namespace(**config)


def args_parser(path_project: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # fmt: off

    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--gpu_id", type=int, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument("--silo_id", type=int, help="silo_id, used for communication")

    parser.add_argument("--dataset_name", type=str, help="name of dataset")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--n_users", type=int, help="number of distinct users: M")
    parser.add_argument("--n_silos", type=int, help="number of silos: S")
    parser.add_argument("--user_dist", type=str, help="users distribution [ uniform-iid, uniform-noniid, zipf-iid, zipf-noniid]")
    parser.add_argument("--silo_dist", type=str, help="silos distribution [ uniform, p, zipf, user-silo-matrix]")
    parser.add_argument("--user_alpha", type=str, help="zipf's parameter for users distribution")
    parser.add_argument("--silo_alpha", type=str, help="zipf's parameter for silos distribution")
    parser.add_argument("--n_labels", type=int, help="number of distinct labels for each user")
    parser.add_argument("--typical_scenaio", type=int, help="typical scenario")

    parser.add_argument("--n_silo_per_round", type=int, help="the silos per round")
    parser.add_argument("--n_total_round", type=int, help="The number of total rounds: R")
    parser.add_argument("--local_epochs", type=int, help="number of local training local_epochs")
    parser.add_argument("--local_learning_rate", type=float, help="local learning rate")
    parser.add_argument("--global_learning_rate", type=float, help="global learning rate")
    parser.add_argument("--local_batch_size", type=int, help="local batch size")
    parser.add_argument("--weight_decay", type=float, help="weight_decay")
    parser.add_argument("--client_optimizer", type=str, help="local of optimizer")

    parser.add_argument("--agg_strategy", type=str, help="aggregation strategy [DEFAULT, SILO-LEVEL-DP, RECORD-LEVEL-DP, ULDP-NAIVE, ULDP-GROUP, ULDP-SGD, ULDP-AVG, ULDP-SGD-w, ULDP-AVG-w]")
    parser.add_argument("--group_k", type=int, help="k (maximum number of user contribution) of group privacy")
    parser.add_argument("--sigma", type=float, help="noise multiplier (Note: std_dev = sigma * clipping_bound))")
    parser.add_argument("--clipping_bound", type=float, help="clipping bound for differential privacy")
    parser.add_argument("--delta", type=float, help="delta for differential privacy")
    parser.add_argument("--sampling_rate_q", type=float, help="sampling rate q for user-level sub-sampling")

    parser.add_argument("--verbose", type=int, help="verbose")
    parser.add_argument("--hyper_parameter_tuning", type=int, help="is hyper-parameter tuning")
    parser.add_argument("--times", type=int, help="times of experiments in different random seeds")
    parser.add_argument("--exp_dist", type=int, help="0 (iid), 1 (non-iid based on zipf)")

    parser.add_argument("--version", type=int, help="used for experimental management")

    # fmt: on

    config = load_config(os.path.join(path_project, DEFAULT_CONFIG_PATH))
    parser.set_defaults(**config)
    args = parser.parse_args()

    assert (
        args.n_silo_per_round <= args.n_silos
    ), "n_silo_per_round should be less than n_silos"

    if args.dataset_name == "isic" and args.gpu_id is None:
        logger.warning("ISIC dataset is too hevay, you should set --gpu_id to use GPU.")

    pretty_print_args(args)

    return args
