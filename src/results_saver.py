import json
import os
import glob
import time
import datetime
import hashlib

from mylogger import logger


RESULTS_DIR = "exp/results"
RESULTS = "results.json"
PARAMETER_DIR = "exp/hyper_parameters"
BEST_PARAMS = "best_params.json"


def load_results(args, path_project):
    hashed_args = args_to_hash(args)
    # print(os.path.join(path_project, RESULTS_DIR, "*_" + hashed_args, RESULTS))
    resutls_file_name_list = glob.glob(
        os.path.join(path_project, RESULTS_DIR, "*_" + hashed_args, RESULTS)
    )
    # print(resutls_file_name_list)
    results_list = []
    for file_path in resutls_file_name_list:
        with open(file_path, "r") as json_file:
            results = json.load(json_file)
        results_list.append(results)
    return results_list, hashed_args


def save_best_params(args, path_project, best_params):
    file_path = os.path.join(path_project, PARAMETER_DIR, BEST_PARAMS)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({}, f)

    with open(file_path, "r") as json_file:
        best_params_dct = json.load(json_file)
    key = str(args)
    best_params_dct[key] = best_params
    with open(file_path, "w") as f:
        json.dump(best_params_dct, f, indent=2)
    logger.info(f"save best_params at key = {key}")


def load_best_params(args, path_project):
    file_path = os.path.join(path_project, PARAMETER_DIR, BEST_PARAMS)
    with open(file_path, "r") as json_file:
        best_params_dct = json.load(json_file)
    key = str(args)
    if key not in best_params_dct:
        logger.info(f"key = {key} is not in best_params")
        return None
    return best_params_dct[key]


def save_resuls(args, path_project, results: dict) -> str:
    hashed_args = args_to_hash(args)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%m%d%H%M%S")
    results_dir = os.path.join(path_project, RESULTS_DIR, timestamp + "_" + hashed_args)
    os.mkdir(results_dir)
    results["args"] = str(args)
    with open(os.path.join(results_dir, RESULTS), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_dir}")


def args_to_hash(args) -> str:
    args_dct = vars(args)
    str_args = str(args_dct)
    hash_obj = hashlib.md5()
    hash_obj.update(str_args.encode())
    return hash_obj.hexdigest()[:10]
