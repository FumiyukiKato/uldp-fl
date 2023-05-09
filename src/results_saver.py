import json
import os
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib

from mylogger import logger


RESULTS_DIR = "exp/results"
RESULTS = "results.json"
PARAMETER_DIR = "exp/hyper_parameters"
BEST_PARAMS = "best_params.json"


def save_best_params(args, path_project, best_params):
    with open(os.path.join(path_project, PARAMETER_DIR, BEST_PARAMS)) as json_file:
        best_params_dct = json.load(json_file)
    key = str(args)
    best_params_dct[key] = best_params
    with open(os.path.join(path_project, PARAMETER_DIR, BEST_PARAMS), "w") as f:
        json.dump(best_params_dct, f, indent=2)
    logger.info(f"save best_params at key = {key}")


def load_best_params(args, path_project):
    with open(os.path.join(path_project, PARAMETER_DIR, BEST_PARAMS)) as json_file:
        best_params = json.load(json_file)
    key = str(args)
    if key not in best_params:
        logger.info(f"key = {key} is not in best_params")
        return None
    return best_params[key]


def save_resuls(args, path_project, results: dict) -> str:
    hashed_args = args_to_hash(args)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%m%d%H%M%S")
    results_dir = os.path.join(path_project, RESULTS_DIR, timestamp + "_" + hashed_args)
    os.mkdir(results_dir)
    results["args"] = str(args)
    with open(os.path.join(results_dir, RESULTS), "w") as f:
        json.dump(results, f, indent=2)

    if results.get("global") is not None:
        if args.agg_strategy in [
            "SILO-LEVEL-DP",
            "RECORD-LEVEL-DP",
            "ULDP-NAIVE",
            "ULDP-GROUP",
            "ULDP-SGD",
            "ULDP-AVG",
        ]:
            privacy_budget_list = results["global"]["privacy_budget"]
            rounds = [round_idx for round_idx, eps, delta in privacy_budget_list]
            eps_list = [eps for round_idx, eps, delta in privacy_budget_list]
            fig, ax = plt.subplots(1, 1)
            sns.lineplot(x=rounds, y=eps_list, marker="o", ax=ax)
            ax.set_xlabel("round", fontsize=18)
            ax.set_ylabel("epsilon", fontsize=18)
            ax.set_title("Epsilon over rounds", fontsize=20)
            fig.savefig(
                os.path.join(results_dir, "global_epsilon.png"),
                dpi=100,
                bbox_inches="tight",
            )

        global_test_list = results["global"]["global_test"]
        rounds = [round_idx for round_idx, test_acc, test_loss in global_test_list]
        test_acc_list = [
            test_acc for round_idx, test_acc, test_loss in global_test_list
        ]
        test_loss_list = [
            test_loss for round_idx, test_acc, test_loss in global_test_list
        ]
        fig, ax = plt.subplots(1, 1)
        sns.lineplot(x=rounds, y=test_acc_list, marker="o", ax=ax)
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylabel("test accuracy", fontsize=18)
        ax.set_title("Test accuracy over rounds", fontsize=20)
        fig.savefig(
            os.path.join(results_dir, "global_test_acc.png"),
            dpi=100,
            bbox_inches="tight",
        )

        fig, ax = plt.subplots(1, 1)
        sns.lineplot(x=rounds, y=test_loss_list, marker="o", ax=ax)
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylabel("test loss", fontsize=18)
        ax.set_title("Test loss over rounds", fontsize=20)
        fig.savefig(
            os.path.join(results_dir, "global_test_loss.png"),
            dpi=100,
            bbox_inches="tight",
        )

    if results.get("local") is not None:
        local_results_per_silos = results["local"]

        local_acc_results = {"round_idx": [], "local_accuracy": [], "silo_id": []}
        local_loss_results = {"round_idx": [], "local_loss": [], "silo_id": []}
        train_time_results = {"round_idx": [], "train_time": [], "silo_id": []}

        for silo_id, local_results in local_results_per_silos.items():
            for round_idx, local_acc, local_loss in local_results["local_test"]:
                local_acc_results["round_idx"].append(round_idx)
                local_acc_results["local_accuracy"].append(local_acc)
                local_acc_results["silo_id"].append(silo_id)

                local_loss_results["round_idx"].append(round_idx)
                local_loss_results["local_loss"].append(local_loss)
                local_loss_results["silo_id"].append(silo_id)

            for round_idx, train_time in local_results["train_time"]:
                train_time_results["round_idx"].append(round_idx)
                train_time_results["train_time"].append(train_time)
                train_time_results["silo_id"].append(silo_id)

        for round_idx in range(args.n_total_round):
            local_acc_of_round = [
                local_acc
                for rid, local_acc in zip(
                    local_acc_results["round_idx"], local_acc_results["local_accuracy"]
                )
                if rid == round_idx
            ]
            local_acc_results["round_idx"].append(round_idx)
            local_acc_results["local_accuracy"].append(
                sum(local_acc_of_round) / len(local_acc_of_round)
            )
            local_acc_results["silo_id"].append("average")
            local_loss_of_round = [
                local_loss
                for rid, local_loss in zip(
                    local_loss_results["round_idx"], local_loss_results["local_loss"]
                )
                if rid == round_idx
            ]
            local_loss_results["round_idx"].append(round_idx)
            local_loss_results["local_loss"].append(
                sum(local_loss_of_round) / len(local_loss_of_round)
            )
            local_loss_results["silo_id"].append("average")
            train_time_of_round = [
                train_time
                for rid, train_time in zip(
                    train_time_results["round_idx"], train_time_results["train_time"]
                )
                if rid == round_idx
            ]
            train_time_results["round_idx"].append(round_idx)
            train_time_results["train_time"].append(
                sum(train_time_of_round) / len(train_time_of_round)
            )
            train_time_results["silo_id"].append("average")

        fig, ax = plt.subplots(1, 1)
        sns.lineplot(
            x="round_idx",
            y="local_accuracy",
            hue="silo_id",
            data=local_acc_results,
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylabel("local accuracy", fontsize=18)
        ax.set_title("Local accuracy over rounds", fontsize=20)
        fig.savefig(
            os.path.join(results_dir, "local_test_acc.png"),
            dpi=100,
            bbox_inches="tight",
        )

        fig, ax = plt.subplots(1, 1)
        sns.lineplot(
            x="round_idx",
            y="local_loss",
            hue="silo_id",
            data=local_loss_results,
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylabel("local loss", fontsize=18)
        ax.set_title("Local loss over rounds", fontsize=20)
        fig.savefig(
            os.path.join(results_dir, "local_test_loss.png"),
            dpi=100,
            bbox_inches="tight",
        )

        fig, ax = plt.subplots(1, 1)
        sns.lineplot(
            x="round_idx",
            y="train_time",
            hue="silo_id",
            data=train_time_results,
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("round", fontsize=18)
        ax.set_ylabel("train time", fontsize=18)
        ax.set_title("Train time over rounds", fontsize=20)
        fig.savefig(
            os.path.join(results_dir, "train_time.png"),
            dpi=100,
            bbox_inches="tight",
        )

    logger.info(f"Results saved to {results_dir}")


def args_to_hash(args) -> str:
    args_dct = vars(args)
    str_args = str(args_dct)
    hash_obj = hashlib.md5()
    hash_obj.update(str_args.encode())
    return hash_obj.hexdigest()[:10]
