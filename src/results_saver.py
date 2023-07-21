import json
import os
import glob
import hashlib
import datetime

from mylogger import logger


RESULTS_DIR = "exp/results"
EXP_RESULTS = "results.json"
HP_DETAIL = "hp_detail.json"
PARAMETER_DIR = "exp"
BEST_PARAMS = "best_params.json"
HP_STORAGE = "hp-search.db"


def load_results(args, path_project, hp: bool = False, hashed_args: str = None):
    if hashed_args is None:
        hashed_args = args_to_hash(args)
    file_name = HP_DETAIL if hp else EXP_RESULTS
    resutls_file_name_list = glob.glob(
        os.path.join(path_project, RESULTS_DIR, hashed_args, file_name)
    )
    print(os.path.join(path_project, RESULTS_DIR, hashed_args, file_name))
    results_list = []
    for file_path in resutls_file_name_list:
        with open(file_path, "r") as json_file:
            results = json.load(json_file)
        results_list.append(results)
    return results_list, hashed_args


def save_resuls(args, path_project, results: dict, hp: bool = False) -> str:
    hashed_args = args_to_hash(args)
    results_dir = os.path.join(path_project, RESULTS_DIR, hashed_args)
    os.makedirs(results_dir, exist_ok=True)
    file_name = HP_DETAIL if hp else EXP_RESULTS
    results["args"] = str(args)
    with open(os.path.join(results_dir, file_name), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_dir}/{file_name}")


def save_one_shot_results(
    args, path_project, results: dict, prefix: str = "sim"
) -> str:
    hashed_args = args_to_hash(args)
    results_dir = os.path.join(path_project, RESULTS_DIR, hashed_args)
    os.makedirs(results_dir, exist_ok=True)
    results["args"] = str(args)
    file_name = (
        prefix + datetime.datetime.now().strftime("-%Y%m%d%H%M%S-") + EXP_RESULTS
    )
    with open(os.path.join(results_dir, file_name), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_dir}/{file_name}")


def save_best_params(args, path_project, best_params):
    file_path = os.path.join(path_project, PARAMETER_DIR, BEST_PARAMS)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({}, f)
    with open(file_path, "r") as json_file:
        best_params_dct = json.load(json_file)

    key = args_to_hash(args)
    best_params_dct[key] = {"params": best_params, "args": str(args)}
    with open(file_path, "w") as f:
        json.dump(best_params_dct, f, indent=2)
    logger.info(f"Save best_params at key = {key}")


def load_best_params(args, path_project):
    file_path = os.path.join(path_project, PARAMETER_DIR, BEST_PARAMS)
    with open(file_path, "r") as json_file:
        best_params_dct = json.load(json_file)

    key = args_to_hash(args)
    if key not in best_params_dct:
        logger.warning(f"key = {key} is not in best_params")
        return None
    return best_params_dct[key]["params"]


def get_storage_path(args, path_project):
    key = args_to_hash(args)
    storage_dir = os.path.join(path_project, "exp", "results", key)
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = os.path.join(storage_dir, HP_STORAGE)
    return key, storage_path


def args_to_hash(args) -> str:
    args_dct: dict = vars(args).copy()
    # excluding hyperparameters
    args_dct.pop("local_epochs")
    args_dct.pop("local_learning_rate")
    args_dct.pop("global_learning_rate")
    args_dct.pop("clipping_bound")

    # excluding unrelated parameters for results
    args_dct.pop("seed")
    args_dct.pop("gpu_id")
    args_dct.pop("silo_id")
    args_dct.pop("verbose")
    args_dct.pop("hyper_parameter_tuning")
    args_dct.pop("times")
    args_dct.pop("dry_run")
    args_dct.pop("secure_w")

    if args_dct["version"] is None or args_dct["version"] == 0:
        args_dct.pop("version")

    str_args = str(args_dct)
    hash_obj = hashlib.md5()
    hash_obj.update(str_args.encode())
    return hash_obj.hexdigest()[:10]


def str_to_namespace(input_str):
    import argparse

    # Remove the 'Namespace(' at the beginning and ')' at the end
    cleaned_str = input_str[10:-1]

    # Convert the string to a dictionary
    str_dict = eval(f"dict({cleaned_str})")
    namespace = argparse.Namespace(**str_dict)
    print(namespace)
    return namespace
