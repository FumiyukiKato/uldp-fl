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
        n_users,
        n_silos,
        n_silo_per_round,
        device,
        base_seed: int,
        strategy: str,
        clipping_bound: Optional[float] = None,
        sigma: Optional[float] = None,
        delta: Optional[float] = None,
        global_learning_rate: Optional[float] = None,
        dataset_name: str = None,
        sampling_rate_q: Optional[float] = None,
    ):
        self.random_state = np.random.RandomState(seed=base_seed + 1000000)
        self.model: nn.Module = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_users = n_users
        self.n_silos = n_silos
        self.n_silo_per_round = n_silo_per_round
        self.device = device
        self.dataset_name = dataset_name
        self.strategy = strategy
        self.global_learning_rate = global_learning_rate
        self.sampling_rate_q = sampling_rate_q
        if self.strategy in [
            "SILO-LEVEL-DP",
            "ULDP-NAIVE",
            "RECORD-LEVEL-DP",
            "ULDP-SGD",
            "ULDP-SGD-w",
            "ULDP-SGD-s",
            "ULDP-SGD-ws",
            "ULDP-AVG",
            "ULDP-AVG-w",
            "ULDP-AVG-s",
            "ULDP-AVG-ws",
        ]:
            from opacus.accountants import RDPAccountant

            assert (
                clipping_bound is not None and sigma is not None and delta is not None
            ), f"Please specify clipping_bound, sigma, delta for {self.strategy}."
            self.clipping_bound = clipping_bound
            self.sigma = sigma
            self.delta = delta
            self.accountant = RDPAccountant()
        elif self.strategy in [
            "RECORD-LEVEL-DP",
            "ULDP-GROUP",
            "ULDP-GROUP-max",
            "ULDP-GROUP-median",
        ]:
            self.delta = delta

        self.model_dict = dict()
        self.n_sample_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.n_silos):
            self.flag_client_model_uploaded_dict[idx] = False

        self.latest_eps = 0.0
        self.results = {"privacy_budget": [], "global_test": [], "local_model_test": []}

    def get_results(self):
        return self.results

    def get_comm_results(self):
        return {}

    def get_global_model_params(self):
        return self.model.state_dict()

    def record_epsilon(self, round_idx):
        if self.strategy in ["SILO-LEVEL-DP"]:
            self.accountant.step(
                noise_multiplier=self.sigma,
                sample_rate=self.n_silo_per_round / self.n_silos,
            )
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in [
            "RECORD-LEVEL-DP",
            "ULDP-GROUP",
            "ULDP-GROUP-median",
            "ULDP-GROUP-max",
        ]:
            eps = self.latest_eps
            self.latest_eps = 0.0
        elif self.strategy in ["ULDP-NAIVE"]:
            self.accountant.step(noise_multiplier=self.sigma, sample_rate=1.0)
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in ["ULDP-SGD", "ULDP-AVG", "ULDP-SGD-w", "ULDP-AVG-w"]:
            self.accountant.step(
                noise_multiplier=self.sigma,
                sample_rate=1.0,
            )
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in [
            "ULDP-SGD-s",
            "ULDP-AVG-s",
            "ULDP-SGD-ws",
            "ULDP-AVG-ws",
        ]:
            self.accountant.step(
                noise_multiplier=self.sigma, sample_rate=self.sampling_rate_q
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
        self.latest_eps = max(self.latest_eps, eps)

    def silo_selection(self) -> List[int]:
        """
        Randomly select n_silo_per_round silos from all silos for each round of FL training.

        Return:
            silo_id_list_in_this_round (list): the list of silo ids to be selected for each round of FL training.
        """

        if self.n_silos == self.n_silo_per_round:
            return np.arange(self.n_silo_per_round).tolist()
        silo_id_list_in_this_round = self.random_state.choice(
            self.n_silos,
            self.n_silo_per_round,
            replace=False,
        )
        logger.debug("Silo selection reuslt: {}".format(silo_id_list_in_this_round))
        return list(silo_id_list_in_this_round)

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
            raw_client_model_or_grad_list.append(self.model_dict[silo_id])

        if self.strategy in [
            "DEFAULT",
            "RECORD-LEVEL-DP",
            "ULDP-GROUP",
            "ULDP-GROUP-max",
            "ULDP-GROUP-median",
            "ULDP-NAIVE",
        ]:
            averaged_param_diff = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list, self.n_silo_per_round
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy in ["ULDP-AVG", "ULDP-AVG-w"]:
            averaged_param_diff = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list, self.n_users * self.n_silo_per_round
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy in ["ULDP-SGD", "ULDP-SGD-w"]:
            averaged_grads = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list, self.n_users * self.n_silo_per_round
            )
            global_weights = self.update_parameters_from_gradients(
                averaged_grads, self.global_learning_rate
            )
        elif self.strategy in ["ULDP-AVG-s", "ULDP-AVG-ws"]:
            averaged_param_diff = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list,
                int(self.n_users * self.n_silo_per_round * self.sampling_rate_q),
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy in ["ULDP-SGD-s", "ULDP-SGD-ws"]:
            averaged_grads = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list,
                int(self.n_users * self.n_silo_per_round * self.sampling_rate_q),
            )
            global_weights = self.update_parameters_from_gradients(
                averaged_grads, self.global_learning_rate
            )
        elif self.strategy in ["SILO-LEVEL-DP"]:
            # https://arxiv.org/abs/1812.06210
            # Usually, this is used in cross-device FL, where the number of participants is large.
            averaged_param_diff = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list, self.n_silo_per_round
            )
            noised_averaged_param_diff = noise_utils.add_global_noise(
                self.model,
                averaged_param_diff,
                random_state=self.random_state,
                std_dev=(self.sigma * self.clipping_bound) / self.n_silo_per_round,
                device=self.device,
            )
            global_weights = self.update_global_weights_from_diff(
                noised_averaged_param_diff
            )
        else:
            raise NotImplementedError(
                "strategy = {} is not implemented".format(self.strategy)
            )

        self.record_epsilon(round_idx)
        return global_weights

    def _test(self, test_data, device, model):
        model.to(device)
        model.eval()

        metrics = {
            "test_metric": 0,
            "test_loss": 0,
            "test_total": 0,
        }

        test_loader = DataLoader(test_data, batch_size=512)

        if self.dataset_name in ["heart_disease", "tcga_brca"]:
            if self.dataset_name == "heart_disease":
                from flamby_utils.heart_disease import (
                    custom_loss,
                    custom_metric,
                )
            elif self.dataset_name == "tcga_brca":
                from flamby_utils.tcga_brca import (
                    custom_loss,
                    custom_metric,
                )

            criterion = custom_loss()
            metric = custom_metric()

            with torch.no_grad():
                y_pred_final = []
                y_true_final = []
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    metrics["test_loss"] += loss.item()
                    y_pred_final.append(y_pred.cpu().numpy())
                    y_true_final.append(y.cpu().numpy())
                    metrics["test_total"] += len(y)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            metrics["test_metric"] = metric(y_true_final, y_pred_final)

        elif self.dataset_name in ["creditcard"]:
            from sklearn.metrics import roc_auc_score

            criterion = nn.CrossEntropyLoss().to(device)

            with torch.no_grad():
                y_pred_final = []
                y_true_final = []
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = model(x)
                    y = y.long()
                    loss = criterion(y_pred, y)
                    metrics["test_loss"] += loss.item()
                    y_pred = y_pred.argmax(dim=1)
                    y_pred_final.append(y_pred.cpu().numpy())
                    y_true_final.append(y.cpu().numpy())
                    metrics["test_total"] += len(y)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            metrics["test_metric"] = roc_auc_score(y_true_final, y_pred_final)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

            with torch.no_grad():
                for x, labels in test_loader:
                    x, labels = x.to(device), labels.to(device)
                    pred = model(x)
                    loss = criterion(pred, labels)
                    metrics["test_loss"] += loss.item()
                    _, predicted = torch.max(pred, 1)
                    metrics["test_metric"] += torch.sum(
                        torch.eq(predicted, labels)
                    ).item()
                    metrics["test_total"] += len(labels)
            metrics["test_metric"] /= metrics["test_total"]

        return metrics

    def test_global(self, round_idx: int, model: nn.Module = None, silo_id: int = None):
        if model is None:
            assert silo_id is None
            model = self.model

        metrics = self._test(self.test_dataset, self.device, model)
        test_metric, n_test_sample, test_loss = (
            metrics["test_metric"],
            metrics["test_total"],
            metrics["test_loss"],
        )

        if silo_id is None:
            self.results["global_test"].append(
                (
                    round_idx,
                    test_metric,
                    test_loss,
                )
            )
            logger.info("|----- Global test result of round %d" % (round_idx))
            if self.dataset_name in ["creditcard"]:
                logger.info(
                    f"\t |----- Test/ROC_AUC: {test_metric} ({n_test_sample}), Test/Loss: {test_loss}"
                )
            else:
                logger.info(
                    f"\t |----- Test/Acc: {test_metric} ({n_test_sample}), Test/Loss: {test_loss}"
                )
        else:
            self.results["local_model_test"].append(
                (
                    round_idx,
                    silo_id,
                    test_metric,
                    test_loss,
                )
            )
            logger.debug(
                "|----- Global test result for SILO %d of round %d"
                % (silo_id, round_idx)
            )
            if self.dataset_name in ["creditcard"]:
                logger.debug(
                    f"\t |----- Test/ROC_AUC: {test_metric} ({n_test_sample}), Test/Loss: {test_loss}"
                )
            else:
                logger.debug(
                    f"\t |----- Test/Acc: {test_metric} ({n_test_sample}), Test/Loss: {test_loss}"
                )
        return test_metric, test_loss

    def update_global_weights_from_diff(
        self, local_weights_diff, learning_rate: float = 1.0
    ) -> Dict:
        """
        Update the parameters of the global model with the difference from the local models.
        """
        global_weights = self.get_global_model_params()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                global_weights[name] += learning_rate * local_weights_diff[name]
        return global_weights

    def update_parameters_from_gradients(
        self, grads, learning_rate: float = 1.0
    ) -> Dict:
        """
        Update the parameters of the global model with the gradients from the local models.

        Input:
            grads (list): aggregated gradients
        Return:
            global_weights (dict): updated global model parameters
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data -= learning_rate * grads[name]
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
