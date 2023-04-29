import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Tuple, OrderedDict
from mylogger import logger


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
        base_seed,
        strategy,
        clipping_bound=None,
        sigma=None,
        delta=None,
    ):
        self.random_state = np.random.RandomState(seed=base_seed + 1000000)
        self.model: nn.Module = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_silos = n_silos
        self.n_silo_per_round = n_silo_per_round
        self.device = device
        self.strategy = strategy
        if self.strategy in ["SILO-LEVEL-DP"]:
            from opacus.accountants import RDPAccountant

            assert (
                clipping_bound is not None and sigma is not None and delta is not None
            ), "Please specify clipping_bound, sigma, delta for SILO-LEVEL-DP."
            self.clipping_bound = clipping_bound
            self.sigma = sigma
            self.delta = delta
            self.accountant = RDPAccountant()
        self.model_dict = dict()
        self.n_sample_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.n_silos):
            self.flag_client_model_uploaded_dict[idx] = False

        self.results = {"privacy_budget": [], "global_test": []}

    def get_results(self):
        return self.results

    def get_comm_results(self):
        return {}

    def get_global_model_params(self):
        return self.model.state_dict()

    def set_global_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def add_local_trained_result(self, silo_id, model_params, n_sample):
        """
        Add the local trained model from a silo to the aggregator.

        Params:
            silo_id (int): the id of the silo.
            model_params (dict): the model parameters of the local trained model.
            n_sample (int): the number of samples used for training the local model.
        """
        model_params = model_params_to_device(model_params, self.device)

        self.model_dict[silo_id] = model_params
        self.n_sample_dict[silo_id] = n_sample
        self.flag_client_model_uploaded_dict[silo_id] = True

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

        # Aggregation w/ or w/o clipping and w/ or w/o weighted averaging
        if self.strategy == "SILO-LEVEL-DP":
            model_list = self.global_clip(
                raw_client_model_or_grad_list, self.clipping_bound
            )
            averaged_params = self.torch_aggregation(model_list)
        elif self.strategy in ["RECORD-LEVEL-DP", "GROUP-DP", "USER-LEVEL-DP"]:
            averaged_params = self.torch_aggregation(raw_client_model_or_grad_list)
        elif self.strategy in ["DEFAULT"]:
            averaged_params = self.torch_weighted_aggregation(
                raw_client_model_or_grad_list
            )
        else:
            raise NotImplementedError(
                "strategy = {} is not implemented".format(self.strategy)
            )

        # Noise addition
        if self.strategy == "SILO-LEVEL-DP":  # https://arxiv.org/abs/1812.06210
            # Usually, this is used in cross-device FL, where the number of participants is large.
            # In cross-silo FL, there is too few participants.
            sample_rate = self.n_silo_per_round / self.n_silos
            self.accountant.step(
                noise_multiplier=self.sigma / self.clipping_bound,
                sample_rate=sample_rate,
            )
            total_epsilon = self.accountant.get_epsilon(self.delta)
            logger.info("Privacy spent: epsilon = {}".format(total_epsilon))
            self.results["privacy_budget"].append(
                (round_idx, total_epsilon, self.delta)
            )
            averaged_params = self._add_global_noise(averaged_params)
        elif self.strategy in [
            "DEFAULT",
            "RECORD-LEVEL-DP",
            "GROUP-DP",
            "USER-LEVEL-DP",
        ]:
            pass
        else:
            raise NotImplementedError(
                "strategy = {} is not implemented".format(self.strategy)
            )

        self.set_global_model_params(averaged_params)
        return averaged_params

    def _add_global_noise(self, grad):
        new_grad = OrderedDict()
        for k in grad.keys():
            new_grad[k] = self._compute_new_grad(grad[k])
        return new_grad

    def _compute_new_grad(self, grad):
        # Gaussian noise
        noise = torch.Tensor(self.random_state.normal(0, self.sigma, size=grad.shape))
        return noise + grad

    def global_clip(
        self,
        raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]],
        clipping_bound: float,
    ):
        """
        Clip the L2-norm of parameters of the local trained models for DP.

        Param:
            raw_client_model_or_grad_list (list): the list of local trained models from the selected silos.
            clipping_bound (float): the L2 clipping bound.
        """
        new_grad_list = []
        for n_sample, local_grad in raw_client_model_or_grad_list:
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(local_grad[k], 2.0) for k in local_grad.keys()]
                ),
                2.0,
            )
            for k in local_grad.keys():
                clip_coef = clipping_bound / (total_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
                for k in local_grad.keys():
                    local_grad[k].mul_(clip_coef_clamped)
            new_grad_list.append((n_sample, local_grad))
        return new_grad_list

    def torch_aggregation(self, raw_grad_list: List):
        """
        Aggregate the local trained models from the selected silos for Pytorch model.

        Params:
            raw_grad_list (list): the list of local trained models from the selected silos.
        Return:
            averaged_params (dict): the averaged model parameters.
        """
        (_, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                _, local_model_params = raw_grad_list[i]
                w = 1 / self.n_silo_per_round
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

    def test_for_all_clients(self, round_idx):
        pass


def model_params_to_device(params_obj, device):
    """
    Change the torch model parameters to the device.
    """
    for key in params_obj.keys():
        params_obj[key] = params_obj[key].to(device)
    return params_obj
