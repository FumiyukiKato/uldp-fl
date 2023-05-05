import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from mylogger import logger
import noise_utils


class Aggregator:
    """
    Aggregator class for Federated Learning.
    """

    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        n_silos,
        n_silo_per_round,
        device,
        base_seed: int,
        strategy: str,
        clipping_bound: Optional[float] = None,
        sigma: Optional[float] = None,
        delta: Optional[float] = None,
        central_learning_rate: Optional[float] = None,
    ):
        self.random_state = np.random.RandomState(seed=base_seed + 1000000)
        self.model: nn.Module = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_silos = n_silos
        self.n_silo_per_round = n_silo_per_round
        self.device = device
        self.strategy = strategy
        if self.strategy in [
            "SILO-LEVEL-DP",
            "ULDP-NAIVE",
            "RECORD-LEVEL-DP",
            "ULDP-SGD",
            "ULDP-AVG",
        ]:
            from opacus.accountants import RDPAccountant

            assert (
                clipping_bound is not None and sigma is not None and delta is not None
            ), "Please specify clipping_bound, sigma, delta for SILO-LEVEL-DP."
            self.clipping_bound = clipping_bound
            self.sigma = sigma
            self.delta = delta
            self.accountant = RDPAccountant()
        elif self.strategy in ["RECORD-LEVEL-DP", "ULDP-GROUP"]:
            self.delta = delta

        if self.strategy in ["ULDP-SGD"]:
            self.central_learning_rate = central_learning_rate

        self.model_dict = dict()
        self.n_sample_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.n_silos):
            self.flag_client_model_uploaded_dict[idx] = False

        self.latest_eps = None
        self.results = {"privacy_budget": [], "global_test": []}

    def get_results(self):
        return self.results

    def get_comm_results(self):
        return {}

    def get_global_model_params(self):
        return self.model.state_dict()

    def set_global_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def record_epsilon(self, round_idx):
        if self.strategy in ["SILO-LEVEL-DP"]:
            self.accountant.step(
                noise_multiplier=self.sigma,
                sample_rate=self.n_silo_per_round / self.n_silos,
            )
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in ["RECORD-LEVEL-DP", "ULDP-GROUP"]:
            eps = self.latest_eps
        elif self.strategy in ["ULDP-NAIVE"]:
            self.accountant.step(noise_multiplier=self.sigma, sample_rate=1.0)
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in ["ULDP-SGD", "ULDP-AVG"]:
            self.accountant.step(
                noise_multiplier=self.sigma,
                sample_rate=1.0,
            )
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in ["DEFAULT"]:
            return
        else:
            raise NotImplementedError(
                "strategy = {} is not implemented".format(self.strategy)
            )

        logger.info("Privacy spent: epsilon = {} (round {})".format(eps, round_idx))
        self.results["privacy_budget"].append((round_idx, eps, self.delta))

    def add_local_trained_result(self, silo_id, model_params, n_sample, eps):
        """
        Add the local trained model from a silo to the aggregator.

        Params:
            silo_id (int): the id of the silo.
            model_params (dict): the model parameters of the local trained model.
            n_sample (int): the number of samples used for training the local model.
            eps (float): the privacy budget used for training the local model.
        """
        model_params = model_params_to_device(model_params, self.device)

        self.model_dict[silo_id] = model_params
        self.n_sample_dict[silo_id] = n_sample
        self.flag_client_model_uploaded_dict[silo_id] = True
        self.latest_eps = eps

    def silo_selection(self):
        """
        Randomly select n_silo_per_round silos from all silos for each round of FL training.

        Return:
            silo_id_list_in_this_round (list): the list of silo ids to be selected for each round of FL training.
        """

        if self.n_silos == self.n_silo_per_round:
            return np.arange(self.n_silo_per_round)
        silo_id_list_in_this_round = self.random_state.choice(
            self.n_silos, self.n_silo_per_round, replace=False
        )
        logger.info("Silo selection reuslt: {}".format(silo_id_list_in_this_round))
        return silo_id_list_in_this_round

    def check_whether_all_receive(self, silo_id_list_in_this_round: List[int]):
        """
        Check whether all silos have uploaded their local trained models to the aggregator.

        Param:
            silo_id_list_in_this_round (list): the list of silo ids to be selected for each round of FL training.
        Return:
            True if all silos have uploaded their local trained models to the aggregator, False otherwise.
        """

        logger.debug("number of silos = {}".format(self.n_silos))
        for silo_id in silo_id_list_in_this_round:
            if not self.flag_client_model_uploaded_dict[silo_id]:
                return False
        for silo_id in range(self.n_silos):
            self.flag_client_model_uploaded_dict[silo_id] = False
        return True

    def aggregate(self, silo_id_list_in_this_round: List[int], round_idx: int):
        """
        Aggregate the local trained models from the selected silos.

        Param:
            silo_id_list_in_this_round (list): the list of silo ids to be selected for each round of FL training.
        Return:
            averaged_params (dict): the averaged model parameters.
        """
        raw_client_model_or_grad_list = []
        for silo_id in silo_id_list_in_this_round:
            raw_client_model_or_grad_list.append(
                (self.n_sample_dict[silo_id], self.model_dict[silo_id])
            )

        if self.strategy in ["SILO-LEVEL-DP"]:
            # https://arxiv.org/abs/1812.06210
            # Usually, this is used in cross-device FL, where the number of participants is large.
            averaged_param_diff = self.torch_aggregation(raw_client_model_or_grad_list)
            noised_averaged_param_diff = noise_utils.add_global_noise(
                averaged_param_diff,
                random_state=self.random_state,
                std_dev=(self.sigma * self.clipping_bound) / self.n_silo_per_round,
            )
            global_weights = self.update_global_weights_from_diff(
                noised_averaged_param_diff
            )
        elif self.strategy in ["RECORD-LEVEL-DP", "ULDP-GROUP"]:
            averaged_param_diff = self.torch_aggregation(raw_client_model_or_grad_list)
            global_weights = self.update_global_weights_from_diff(averaged_param_diff)
        elif self.strategy in ["ULDP-NAIVE"]:
            averaged_param_diff = self.torch_aggregation(raw_client_model_or_grad_list)
            global_weights = self.update_global_weights_from_diff(averaged_param_diff)
        elif self.strategy in ["ULDP-SGD"]:
            averaged_grads = self.torch_aggregation(raw_client_model_or_grad_list)
            total_norm = torch.norm(
                torch.stack([torch.norm(g, 2.0) for g in averaged_grads]), 2.0
            )
            logger.info(
                "l2 norm of averaged_grads = {}, learning_rate = {}, Delta = {}".format(
                    total_norm,
                    self.central_learning_rate,
                    total_norm * self.central_learning_rate,
                )
            )
            global_weights = self.update_parameters_from_gradients(averaged_grads)
        elif self.strategy in ["ULDP-AVG"]:
            averaged_param_diff = self.torch_aggregation(raw_client_model_or_grad_list)
            total_norm = torch.norm(
                torch.stack(
                    [
                        torch.norm(averaged_param_diff[k], 2.0)
                        for k in averaged_param_diff.keys()
                    ]
                ),
                2.0,
            )
            logger.info("l2 norm of averaged_param_diff = {}".format(total_norm))
            global_weights = self.update_global_weights_from_diff(averaged_param_diff)
        elif self.strategy in ["DEFAULT"]:
            averaged_param_diff = self.torch_aggregation(raw_client_model_or_grad_list)
            global_weights = self.update_global_weights_from_diff(averaged_param_diff)
        else:
            raise NotImplementedError(
                "strategy = {} is not implemented".format(self.strategy)
            )

        self.record_epsilon(round_idx)
        return global_weights

    def torch_aggregation(self, raw_grad_list: List) -> Dict:
        """
        Aggregate the local trained models from the selected silos for Pytorch model.

        Params:
            raw_grad_list (list): the list of local trained models from the selected silos.
        Return:
            averaged_params (dict): the averaged model parameters.
        """
        (_, avg_params) = raw_grad_list[0]
        w = 1.0 / self.n_silo_per_round

        if type(avg_params) == list:
            for i in range(0, len(avg_params)):
                for j in range(0, len(raw_grad_list)):
                    _, local_model_params = raw_grad_list[j]
                    if j == 0:
                        avg_params[i] = local_model_params[i] * w
                    else:
                        avg_params[i] += local_model_params[i] * w
            return avg_params
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                _, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
        return avg_params

    def torch_weighted_aggregation(self, raw_grad_list: List):
        """
        Aggregate the local trained models from the selected silos
        for Pytorch model with weights based on local sample size.

        Params:
            raw_grad_list (list): the list of local trained models from the selected silos.
        Return:
            averaged_params (dict): the averaged model parameters.
        """
        n_total_sample = 0
        for i in range(len(raw_grad_list)):
            n_local_sample, _ = raw_grad_list[i]
            n_total_sample += n_local_sample

        (_, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                n_local_sample, local_model_params = raw_grad_list[i]
                w = n_local_sample / n_total_sample
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
        return avg_params

    def _test(self, test_data, device):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        test_loader = DataLoader(test_data, batch_size=10)
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for x, labels in test_loader:
                x, labels = x.to(device), labels.to(device)
                pred = model(x)

                loss = criterion(pred, labels)
                metrics["test_loss"] += loss.item()

                _, predicted = torch.max(pred, 1)
                metrics["test_correct"] += torch.sum(torch.eq(predicted, labels)).item()

                metrics["test_total"] += len(labels)

        return metrics

    def test_global(self, round_idx: int):
        # test data
        metrics = self._test(self.test_dataset, self.device)

        test_tot_correct, n_test_sample, test_loss = (
            metrics["test_correct"],
            metrics["test_total"],
            metrics["test_loss"],
        )

        test_acc = test_tot_correct / n_test_sample
        self.results["global_test"].append(
            (
                round_idx,
                test_acc,
                test_loss,
            )
        )

        logger.info("|----- Global test result of round %d" % (round_idx))
        logger.info(
            f"\t |----- Test/Acc: {test_acc} ({test_tot_correct} / {n_test_sample}), Test/Loss: {test_loss}"
        )
        return test_acc, test_loss

    def update_global_weights_from_diff(self, local_weights_diff) -> Dict:
        """
        Update the parameters of the global model with the difference from the local models.
        """
        global_weights = self.get_global_model_params()
        for key in global_weights.keys():
            global_weights[key] += local_weights_diff[key]
        return global_weights

    def update_parameters_from_gradients(self, grads) -> Dict:
        """
        Update the parameters of the global model with the gradients from the local models.

        Input:
            grads (list): aggregated gradients
        Return:
            global_weights (dict): updated global model parameters
        """
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), grads):
                param -= self.central_learning_rate * grad
        return self.model.state_dict()


def model_params_to_device(params_obj, device):
    """
    Change the torch model parameters to the device.
    """
    if type(params_obj) == list:
        for i in range(len(params_obj)):
            params_obj[i] = params_obj[i].to(device)
        return params_obj
    for key in params_obj.keys():
        params_obj[key] = params_obj[key].to(device)
    return params_obj
