import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Optional, OrderedDict, Union, Tuple
from opacus.accountants import RDPAccountant
from dataset import CREDITCARD, HEART_DISEASE, TCGA_BRCA
from method_group import (
    METHOD_GROUP_AGGREGATOR_PRIVACY_ACCOUNTING,
    METHOD_GROUP_AVG,
    METHOD_GROUP_DP,
    METHOD_GROUP_GRADIENT,
    METHOD_GROUP_NO_SAMPLING,
    METHOD_GROUP_PARAMETER_DIFF,
    METHOD_GROUP_SAMPLING,
    METHOD_GROUP_WEIGHTS,
    METHOD_NO_DP_ACCOUNTING,
    METHOD_PULDP_AVG,
    METHOD_GROUP_ONLINE_OPTIMIZATION,
    METHOD_PULDP_AVG_ONLINE,
    METHOD_PULDP_AVG_ONLINE_TRAIN,
    METHOD_SILO_LEVEL_DP,
    METHOD_ULDP_NAIVE,
)

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
        test_dataset: List[Tuple[torch.Tensor, int]],
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
        validation_ratio: float = 0.0,
        n_total_round: Optional[int] = None,
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
        self.n_total_round = n_total_round
        if self.strategy in METHOD_GROUP_AGGREGATOR_PRIVACY_ACCOUNTING:
            assert (
                clipping_bound is not None and sigma is not None and delta is not None
            ), f"Please specify clipping_bound, sigma, delta for {self.strategy}."
            self.clipping_bound = clipping_bound
            self.sigma = sigma
            self.delta = delta
            self.accountant = RDPAccountant()
        elif self.strategy in METHOD_GROUP_DP:
            self.delta = delta

        self.model_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.n_silos):
            self.flag_client_model_uploaded_dict[idx] = False

        self.latest_eps = 0.0
        self.results = {"privacy_budget": [], "global_test": [], "local_model_test": []}

        if self.strategy == METHOD_PULDP_AVG_ONLINE:
            assert (
                validation_ratio > 0.0
            ), "Please set validation_ratio > 0 for Test Loss-based online."

        if validation_ratio > 0.0:
            n_test = len(test_dataset)
            n_validation = int(n_test * validation_ratio)
            n_test = n_test - n_validation
            test_dataset, validation_dataset = random_split(
                test_dataset,
                [n_test, n_validation],
                torch.Generator().manual_seed(self.random_state.randint(2**32 - 1)),
            )
            self.test_dataset = test_dataset
            self.validation_dataset = validation_dataset
            self.results["global_valid"] = []

        if self.strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
            self.model_dict_for_optimization = {
                silo_id: {} for silo_id in range(self.n_silos)
            }
            self.results["local_loss_diff"] = []

    def get_results(self):
        return self.results

    def get_comm_results(self):
        return {}

    def get_global_model_params(self):
        return self.model.state_dict()

    def set_epsilon_groups(self, epsilon_groups: Dict[float, np.ndarray]):
        self.epsilon_groups = epsilon_groups
        self.accountant_dct = {eps_u: RDPAccountant() for eps_u in epsilon_groups}

    def record_epsilon(self, round_idx):
        if self.strategy == METHOD_SILO_LEVEL_DP:  #
            self.accountant.step(
                noise_multiplier=self.sigma,
                sample_rate=self.n_silo_per_round / self.n_silos,
            )
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in METHOD_GROUP_DP.difference(
            METHOD_GROUP_AGGREGATOR_PRIVACY_ACCOUNTING
        ):
            eps = self.latest_eps
            self.latest_eps = 0.0
        elif self.strategy == METHOD_ULDP_NAIVE:
            self.accountant.step(noise_multiplier=self.sigma, sample_rate=1.0)
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in METHOD_GROUP_NO_SAMPLING:
            self.accountant.step(
                noise_multiplier=self.sigma,
                sample_rate=1.0,
            )
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in METHOD_GROUP_SAMPLING.difference({METHOD_PULDP_AVG}):
            self.accountant.step(
                noise_multiplier=self.sigma, sample_rate=self.sampling_rate_q
            )
            eps = self.accountant.get_epsilon(self.delta)
        elif self.strategy in METHOD_NO_DP_ACCOUNTING:
            return
        else:
            raise NotImplementedError(
                "strategy = {} is not implemented".format(self.strategy)
            )

        logger.info("Privacy spent: epsilon = {} (round {})".format(eps, round_idx))
        self.results["privacy_budget"].append((round_idx, eps, self.delta))

    def add_local_trained_result(self, silo_id, model_params, eps):
        """
        Add the local trained model from a silo to the aggregator.

        Params:
            silo_id (int): the id of the silo.
            model_params (dict): the model parameters of the local trained model.
            eps (float): the privacy budget used for training the local model.
        """
        model_params = model_params_to_device(model_params, self.device)

        self.model_dict[silo_id] = model_params
        self.flag_client_model_uploaded_dict[silo_id] = True
        self.latest_eps = max(self.latest_eps, eps)

    def add_local_trained_result_with_online_optimization(
        self,
        silo_id,
        model_params,
        eps,
        local_diff_dct: Dict[float, float],
    ):
        """
        For online HP Optimization, we need to store the local loss diff for each eps groups.
        """
        self.add_local_trained_result(silo_id, model_params, eps)
        for eps_u, local_diff in local_diff_dct.items():
            self.model_dict_for_optimization[silo_id][eps_u] = local_diff

    def add_local_trained_result_with_static_optimization(
        self, silo_id, model_params_dct: Dict[Union[float, str], OrderedDict], eps
    ):
        """For static HP Optimization, we need to store the local trained models for each eps groups."""
        for eps_u, model_params in model_params_dct.items():
            if eps_u == "default":  # DEFAULT_NAME in local_trainer.py
                model_params = model_params_to_device(model_params, self.device)
                self.model_dict[silo_id] = model_params
            else:
                original_model_params = model_params_to_device(
                    model_params[0], self.device
                )
                stepped_model_params = model_params_to_device(
                    model_params[1], self.device
                )
                self.model_dict_for_optimization[silo_id][eps_u] = (
                    original_model_params,
                    stepped_model_params,
                )
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
        logger.debug("Silo selection result: {}".format(silo_id_list_in_this_round))
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

        if self.strategy in METHOD_GROUP_PARAMETER_DIFF.difference(
            METHOD_GROUP_NO_SAMPLING
        ):
            averaged_param_diff = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list, self.n_silo_per_round
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy in METHOD_GROUP_AVG.difference(METHOD_GROUP_SAMPLING):
            averaged_param_diff = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list, self.n_users * self.n_silo_per_round
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy in METHOD_GROUP_GRADIENT.difference(METHOD_GROUP_SAMPLING):
            averaged_grads = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list, self.n_users * self.n_silo_per_round
            )
            global_weights = self.update_parameters_from_gradients(
                averaged_grads, self.global_learning_rate
            )
        elif self.strategy in METHOD_GROUP_WEIGHTS.difference(METHOD_GROUP_NO_SAMPLING):
            averaged_param_diff = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list,
                int(self.n_users * self.n_silo_per_round * self.sampling_rate_q),
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy in METHOD_GROUP_GRADIENT.intersection(METHOD_GROUP_SAMPLING):
            averaged_grads = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list,
                int(self.n_users * self.n_silo_per_round * self.sampling_rate_q),
            )
            global_weights = self.update_parameters_from_gradients(
                averaged_grads, self.global_learning_rate
            )
        elif self.strategy == METHOD_SILO_LEVEL_DP:
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

        if self.dataset_name in [HEART_DISEASE, TCGA_BRCA]:
            if self.dataset_name == HEART_DISEASE:
                from flamby_utils.heart_disease import (
                    custom_loss,
                    custom_metric,
                )
            elif self.dataset_name == TCGA_BRCA:
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

        elif self.dataset_name == CREDITCARD:
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

    def test_global(
        self,
        round_idx: int,
        model: nn.Module = None,
        silo_id: int = None,
        is_validation: bool = False,
    ):
        if model is None:
            assert silo_id is None
            model = self.model

        if is_validation:
            assert hasattr(
                self, "validation_dataset"
            ), "Please set validation=True in the constructor."
            metrics = self._test(self.validation_dataset, self.device, model)
            symbol = "valid"
        else:
            metrics = self._test(self.test_dataset, self.device, model)
            symbol = "test"

        test_metric, n_test_sample, test_loss = (
            metrics["test_metric"],
            metrics["test_total"],
            metrics["test_loss"],
        )

        if silo_id is None:
            self.results[f"global_{symbol}"].append(
                (
                    round_idx,
                    test_metric,
                    test_loss,
                )
            )
            logger.info(f"|----- Global {symbol} result of round {round_idx}")
            if self.dataset_name == CREDITCARD:
                logger.info(
                    f"\t |----- Test/ROC_AUC: {test_metric} ({n_test_sample}), Test/Loss: {test_loss}"
                )
            else:
                logger.info(
                    f"\t |----- Test/Acc: {test_metric} ({n_test_sample}), Test/Loss: {test_loss}"
                )
        else:
            self.results[f"local_model_{symbol}"].append(
                (
                    round_idx,
                    silo_id,
                    test_metric,
                    test_loss,
                )
            )
            logger.debug(
                f"|----- Global {symbol} result for SILO {silo_id} of round {round_idx}"
            )
            if self.dataset_name == CREDITCARD:
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

    def compute_loss_diff(
        self,
        silo_id_list_in_this_round=None,
        q_u_list=None,
        stepped_q_u_list=None,
    ) -> Dict[float, float]:
        """Compute the loss difference for each epsilon group to
        approximately perform Finite Difference Method for HP optimization.
        """
        diff_dct = {}

        if self.strategy == METHOD_PULDP_AVG_ONLINE_TRAIN:
            for local_loss_diff_dct in self.model_dict_for_optimization.values():
                for eps_u, local_loss_diff in local_loss_diff_dct.items():
                    if eps_u not in diff_dct:
                        diff_dct[eps_u] = 0.0
                    diff_dct[eps_u] += local_loss_diff
            return diff_dct

        original_model = copy.deepcopy(self.model)
        for eps_u, eps_user_ids in self.epsilon_groups.items():
            raw_client_model_or_grad_list = []
            for silo_id in silo_id_list_in_this_round:
                raw_client_model_or_grad_list.append(
                    self.model_dict_for_optimization[silo_id][eps_u][0]
                )
            q_list = q_u_list[eps_user_ids]
            sampling_rate_q = np.mean(q_list)
            averaged_param_diff = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list,
                int(len(eps_user_ids) * self.n_silo_per_round * sampling_rate_q),
            )
            self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
            metrics = self._test(self.test_dataset, self.device, self.model)
            original_test_loss = metrics["test_loss"]

            self.model = copy.deepcopy(original_model)
            raw_client_model_or_grad_list = []
            for silo_id in silo_id_list_in_this_round:
                raw_client_model_or_grad_list.append(
                    self.model_dict_for_optimization[silo_id][eps_u][1]
                )
            q_list = stepped_q_u_list[eps_user_ids]
            sampling_rate_q = np.mean(q_list)
            averaged_param_diff = noise_utils.torch_aggregation(
                raw_client_model_or_grad_list,
                int(len(eps_user_ids) * self.n_silo_per_round * sampling_rate_q),
            )
            self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
            metrics = self._test(self.test_dataset, self.device, self.model)
            stepped_test_loss = metrics["test_loss"]

            diff = stepped_test_loss - original_test_loss
            diff_dct[eps_u] = diff
            logger.info("eps_u = {}, diff = {}".format(eps_u, diff))

        self.model = original_model
        return diff_dct

    def consume_dp_for_train_loss_metric(
        self,
        q_u_list,
        stepped_q_u_list,
        round_idx: int,
    ):
        # train_loss_metric is approximated metric instead of the real train loss
        # it needs to be bounded by 1
        for eps_u, eps_user_ids in self.epsilon_groups.items():
            q_u = q_u_list[eps_user_ids[0]]
            stepped_q_u = stepped_q_u_list[eps_user_ids[0]]
            (
                noise_multiplier,
                _,
            ) = noise_utils.get_noise_multiplier_with_history(
                q_u,
                self.delta,
                epsilon_u=eps_u,
                total_round=self.n_total_round * 3,
                current_round=round_idx * 3 + 1,
                current_accountant=self.accountant_dct[eps_u],
            )
            (
                stepped_noise_multiplier,
                _,
            ) = noise_utils.get_noise_multiplier_with_history(
                stepped_q_u,
                self.delta,
                epsilon_u=eps_u,
                total_round=self.n_total_round * 3,
                current_round=round_idx * 3 + 2,
                current_accountant=self.accountant_dct[eps_u],
            )
            self.accountant_dct[eps_u].step(
                noise_multiplier=noise_multiplier,
                sample_rate=q_u,
            )
            self.accountant_dct[eps_u].step(
                noise_multiplier=stepped_noise_multiplier,
                sample_rate=stepped_q_u,
            )

    def consume_dp_for_model_optimization(self, q_u_list, C_u_list):
        for eps_u, eps_user_ids in self.epsilon_groups.items():
            q_u = q_u_list[eps_user_ids[0]]
            C_u = C_u_list[eps_user_ids[0]]
            self.accountant_dct[eps_u].step(
                noise_multiplier=self.sigma / C_u,
                sample_rate=q_u,
            )


def model_params_to_device(params_obj, device):
    """
    Change the torch model parameters to the device.
    """
    if type(params_obj) is list:
        for i in range(len(params_obj)):
            params_obj[i] = params_obj[i].to(device)
        return params_obj
    for key in params_obj.keys():
        params_obj[key] = params_obj[key].to(device)
    return params_obj
