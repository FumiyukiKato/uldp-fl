from collections import OrderedDict
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader

from mylogger import logger
import noise_utils
import copy


class ClassificationTrainer:
    """
    Trainer for classification models.
    """

    def __init__(
        self,
        base_seed: int,
        model,
        silo_id: int,
        device: str,
        agg_strategy: str,
        local_train_dataset: List[Tuple[torch.Tensor, int]],
        local_test_dataset: List[Tuple[torch.Tensor, int]],
        user_histogram: Optional[Dict[int, int]],
        user_ids_of_local_train_dataset: Optional[List[int]],
        client_optimizer: str = "sgd",
        local_learning_rate: float = 0.001,
        local_batch_size: int = 1,
        weight_decay: float = 0.001,
        local_epochs: int = 5,
        local_delta: Optional[float] = None,
        local_sigma: Optional[float] = None,
        local_clipping_bound: Optional[float] = None,
        group_k: Optional[int] = None,
        user_weights: Optional[Dict[int, float]] = None,
        n_silo_per_round: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ):
        self.random_state = np.random.RandomState(seed=base_seed + silo_id + 1)
        self.model: nn.Module = model
        self.silo_id = silo_id
        self.device = device
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.user_weights = user_weights
        self.n_silo_per_round = n_silo_per_round
        self.local_learning_rate = local_learning_rate
        self.dataset_name = dataset_name

        self.results = {"local_test": [], "train_time": [], "epsilon": []}

        self.agg_strategy = agg_strategy
        if self.agg_strategy in [
            "RECORD-LEVEL-DP",
            "ULDP-GROUP",
            "ULDP-GROUP-max",
            "ULDP-GROUP-median",
        ]:
            assert client_optimizer == "sgd"
            from opacus import PrivacyEngine

            self.privacy_engine = PrivacyEngine(accountant="rdp")
            self.local_delta = local_delta
            self.local_sigma = local_sigma
            self.local_clipping_bound = local_clipping_bound
            self.group_k = group_k
        elif self.agg_strategy in [
            "ULDP-NAIVE",
            "SILO-LEVEL-DP",
            "ULDP-SGD",
            "ULDP-SGD-w",
            "ULDP-SGD-s",
            "ULDP-SGD-ws",
            "ULDP-AVG",
            "ULDP-AVG-w",
            "ULDP-AVG-s",
            "ULDP-AVG-ws",
        ]:
            self.local_sigma = local_sigma
            self.local_clipping_bound = local_clipping_bound

        if self.dataset_name == "heart_disease":
            from flamby_utils.heart_disease import (
                custom_loss,
                custom_optimizer,
                custom_metric,
            )

            self.criterion = custom_loss()
            self.optimizer = custom_optimizer(self.model, self.local_learning_rate)
            self.metric = custom_metric()
        elif self.dataset_name == "tcga_brca":
            from flamby_utils.tcga_brca import (
                custom_loss,
                custom_optimizer,
                custom_metric,
            )

            self.criterion = custom_loss()
            self.optimizer = custom_optimizer(self.model, self.local_learning_rate)
            self.metric = custom_metric()
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            if client_optimizer == "sgd":
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=local_learning_rate,
                )
            elif client_optimizer == "adam":
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=local_learning_rate,
                    weight_decay=weight_decay,
                    amsgrad=True,
                )
            else:
                raise ValueError("Unknown client optimizer")

        self.train_loader = DataLoader(
            local_train_dataset, batch_size=local_batch_size, shuffle=True
        )
        self.test_loader = DataLoader(local_test_dataset)
        self.user_histogram = user_histogram
        self.user_ids_of_local_train_dataset = user_ids_of_local_train_dataset
        self.distinct_users = list(set(user_ids_of_local_train_dataset))
        if self.agg_strategy in [
            "ULDP-SGD",
            "ULDP-SGD-w",
            "ULDP-AVG",
            "ULDP-AVG-w",
            "ULDP-SGD-s",
            "ULDP-AVG-s",
            "ULDP-AVG-ws",
            "ULDP-SGD-ws",
        ]:
            self.user_level_data_loader = self.make_user_level_data_loader()

    def get_results(self):
        return self.results

    def get_comm_results(self):
        return {}

    def get_model_params(self):
        return self.model.state_dict()

    def get_torch_manual_seed(self):
        return self.random_state.randint(9223372036854775807)

    def get_latest_epsilon(self):
        if len(self.results["epsilon"]) == 0:
            return 0.0
        return self.results["epsilon"][-1][1]

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def set_user_weights(self, user_weights: Dict[int, float]):
        self.user_weights = user_weights

    def set_group_k(self, group_k: int):
        self.group_k = group_k

    def make_user_level_data_loader(self) -> List[Tuple[int, DataLoader]]:
        shuffled_train_data_indices = np.arange(len(self.train_loader.dataset))
        self.random_state.shuffle(shuffled_train_data_indices)
        data_per_users: Dict[int, List] = {}
        for idx in shuffled_train_data_indices:
            user_id = self.user_ids_of_local_train_dataset[idx]
            data = self.train_loader.dataset[idx]
            if user_id not in data_per_users:
                data_per_users[user_id] = []
            data_per_users[user_id].append(data)

        new_train_loader_list = []

        shuffled_distinct_user_indices = np.arange(len(self.distinct_users))
        self.random_state.shuffle(shuffled_distinct_user_indices)
        for idx in shuffled_distinct_user_indices:
            user_id = self.distinct_users[idx]
            new_train_loader_list.append(
                (
                    user_id,
                    DataLoader(
                        data_per_users[user_id],
                        batch_size=self.local_batch_size,
                        shuffle=True,
                    ),
                )
            )
        return new_train_loader_list

    def bound_user_contributions(self, bounded_user_histogram):
        new_local_train_dataset = []
        new_local_test_dataset = self.test_loader.dataset
        new_user_ids_of_local_train_dataset = []

        user_counter = {}
        remove_counter = 0
        for user_id, data in zip(
            self.user_ids_of_local_train_dataset, self.train_loader.dataset
        ):
            if user_id not in user_counter:
                user_counter[user_id] = 1
            else:
                user_counter[user_id] += 1
            if (
                user_id in bounded_user_histogram
                and user_counter[user_id] <= bounded_user_histogram[user_id]
            ):
                new_local_train_dataset.append(data)
                new_user_ids_of_local_train_dataset.append(user_id)
            else:
                remove_counter += 1
                if self.dataset_name not in ["heart_disease", "tcga_brca"]:
                    new_local_test_dataset.append(data)
        logger.debug("{} data is removed from training dataset".format(remove_counter))

        if len(new_local_train_dataset) <= 0:
            logger.error("No training data is left after bounding user contributions")
            raise AssertionError(
                "No training data is left after bounding user contributions"
            )
        self.train_loader = DataLoader(
            new_local_train_dataset, batch_size=self.local_batch_size, shuffle=True
        )
        self.test_loader = DataLoader(new_local_test_dataset)
        self.user_ids_of_local_train_dataset = new_user_ids_of_local_train_dataset
        self.distinct_users = list(set(new_user_ids_of_local_train_dataset))

    def train(
        self, global_round_index: int, loss_callback: Callable = lambda loss: None
    ):
        """
        Train the model on the local dataset.
        """
        tick = time.time()

        model = self.model
        model.to(self.device)
        model.train()
        global_weights = copy.deepcopy(self.get_model_params())

        torch.manual_seed(self.get_torch_manual_seed())

        train_loader = self.train_loader
        optimizer = self.optimizer
        criterion = self.criterion

        # Optimization step like CALCULATE GRADIENTS
        if self.agg_strategy in [
            "RECORD-LEVEL-DP",
            "ULDP-GROUP",
            "ULDP-GROUP-max",
            "ULDP-GROUP-median",
        ]:
            noise_generator = torch.Generator(device=self.device).manual_seed(
                self.get_torch_manual_seed()
            )
            model, optimizer, train_loader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                max_grad_norm=self.local_clipping_bound,
                noise_multiplier=self.local_sigma,
                noise_generator=noise_generator,
            )

        if self.agg_strategy in ["ULDP-SGD", "ULDP-SGD-w", "ULDP-SGD-s", "ULDP-SGD-ws"]:
            grads_list = []  # TODO: memory optimization (use online aggregation)
            for user_id, user_train_loader in self.user_level_data_loader:
                if (
                    self.user_weights[user_id] <= 0.0
                ):  # for efficiency, if w is encrypted for DDP, it can't work
                    continue
                user_avg_grad = OrderedDict()
                for name, param in model.named_parameters():
                    user_avg_grad[name] = torch.zeros_like(param.data)

                for x, labels in user_train_loader:
                    x, labels = x.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    log_probs = model(x)
                    if self.dataset_name in ["creditcard"]:
                        labels = labels.long()
                    loss = criterion(log_probs, labels)
                    loss_callback(loss)
                    loss.backward()
                    # Don't optimize (i.e., Don't call step())

                    for name, param in model.named_parameters():
                        # Due to different batch size for each user
                        user_avg_grad[name] += param.grad / len(x)

                clipped_grads = noise_utils.global_clip(
                    model, user_avg_grad, self.local_clipping_bound
                )
                weighted_clipped_grads = noise_utils.multiple_weights(
                    model, clipped_grads, self.user_weights[user_id]
                )
                grads_list.append(weighted_clipped_grads)

            # calculate the average gradient
            if len(grads_list) <= 0:
                avg_grads = OrderedDict()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        avg_grads[name] = torch.zeros_like(param.data)
            else:
                avg_grads = noise_utils.torch_aggregation(grads_list, 1.0)
            noisy_avg_grads = noise_utils.add_global_noise(
                model,
                avg_grads,
                self.random_state,
                self.local_sigma
                * self.local_clipping_bound
                / np.sqrt(self.n_silo_per_round),
                device=self.device,
            )

        elif self.agg_strategy in [
            "ULDP-AVG",
            "ULDP-AVG-w",
            "ULDP-AVG-s",
            "ULDP-AVG-ws",
        ]:
            # If suddenly becomes unstable, skip the update graident (call step()) for now.
            def loss_callback(loss):
                if torch.isnan(loss):
                    logger.warn("loss is nan: skipping")
                    return True
                return False

            weights_diff_list = []  # TODO: memory optimization (use online aggregation)
            for user_id, user_train_loader in self.user_level_data_loader:
                if (
                    self.user_weights[user_id] <= 0.0
                ):  # for efficiency, if w is encrypted for DDP, it can't work
                    continue
                model_u = copy.deepcopy(model)
                optimizer_u = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model_u.parameters()),
                    lr=self.local_learning_rate,
                )
                for epoch in range(self.local_epochs):
                    batch_loss = []
                    for x, labels in user_train_loader:
                        x, labels = x.to(self.device), labels.to(self.device)
                        optimizer_u.zero_grad()
                        log_probs = model_u(x)
                        if self.dataset_name in ["creditcard"]:
                            labels = labels.long()
                        loss = criterion(log_probs, labels)
                        if loss_callback(loss):
                            continue
                        loss.backward()
                        optimizer_u.step()
                        batch_loss.append(loss.item())

                    # logger.debug(
                    #     "Silo Id = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    #         self.silo_id, epoch, sum(batch_loss) / len(batch_loss)
                    #     )
                    # )
                weights = model_u.state_dict()
                weights_diff = self.diff_weights(global_weights, weights)
                clipped_weights_diff = noise_utils.global_clip(
                    model_u, weights_diff, self.local_clipping_bound
                )
                weighted_clipped_weights_diff = noise_utils.multiple_weights(
                    model_u, clipped_weights_diff, self.user_weights[user_id]
                )
                weights_diff_list.append(weighted_clipped_weights_diff)

            if len(weights_diff_list) <= 0:
                avg_weights_diff = OrderedDict()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        avg_weights_diff[name] = torch.zeros_like(param.data)
            else:
                avg_weights_diff = noise_utils.torch_aggregation(weights_diff_list, 1.0)
            noisy_avg_weights_diff = noise_utils.add_global_noise(
                model,
                avg_weights_diff,
                self.random_state,
                self.local_sigma
                * self.local_clipping_bound
                / np.sqrt(self.n_silo_per_round),
                device=self.device,
            )

        else:
            for epoch in range(self.local_epochs):
                batch_loss = []
                for x, labels in train_loader:
                    if len(x) == 0:  # this is possible in poisson sampling in DP-SGD
                        continue
                    x, labels = x.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    log_probs = model(x)
                    if self.dataset_name in ["creditcard"]:
                        labels = labels.long()
                    loss = criterion(log_probs, labels)
                    loss_callback(loss)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                if len(batch_loss) == 0:
                    logger.debug(
                        "Silo Id = {}\tEpoch: {}\t NO SAMPLES".format(
                            self.silo_id, epoch
                        )
                    )
                    continue
                # logger.debug(
                #     "Silo Id = {}\tEpoch: {}\tLoss: {:.6f}".format(
                #         self.silo_id, epoch, sum(batch_loss) / len(batch_loss)
                #     )
                # )

            weights = self.get_model_params()
            weights_diff = self.diff_weights(global_weights, weights)

        train_time = time.time() - tick
        logger.debug("Train/Time : %s", train_time)
        self.results["train_time"].append((global_round_index, train_time))

        # Post-process step
        if self.agg_strategy in ["RECORD-LEVEL-DP"]:
            model.remove_hooks()
            eps = self.privacy_engine.get_epsilon(delta=self.local_delta)
            logger.debug(
                "Silo Id = {}\tEpsilon: {:.6f} (delta: {:6f})".format(
                    self.silo_id, eps, self.local_delta
                )
            )
            self.results["epsilon"].append((global_round_index, eps))
            return weights_diff, len(train_loader)
        elif self.agg_strategy in [
            "ULDP-GROUP",
            "ULDP-GROUP-max",
            "ULDP-GROUP-median",
        ]:
            model.remove_hooks()
            # group_eps_from_rdp, opt_alpha = noise_utils.get_group_privacy_spent(
            #     group_k=self.group_k,
            #     accountant_history=self.privacy_engine.accountant.history,
            #     delta=self.local_delta,
            # )
            # (
            #     group_eps_from_normal_dp_conversion,
            #     delta,
            # ) = noise_utils.get_normal_group_privacy_spent(
            #     group_k=self.group_k,
            #     accountant_history=self.privacy_engine.accountant.history,
            #     delta=self.local_delta,
            # )
            # logger.debug(
            #     "Silo Id = {}\t (Group-Privacy) Epsilon: {:.5f} (delta: {:8f}) (RDP Epsilon: {:.5f})".format(
            #         self.silo_id,
            #         group_eps_from_normal_dp_conversion,
            #         delta,
            #         group_eps_from_rdp,
            #     )
            # )
            # self.results["epsilon"].append(
            #     (
            #         global_round_index,
            #         group_eps_from_normal_dp_conversion,
            #         group_eps_from_rdp,
            #     )
            # )
            return weights_diff, len(train_loader)
        elif self.agg_strategy in ["ULDP-NAIVE"]:
            clipped_weights_diff = noise_utils.global_clip(
                self.model, weights_diff, self.local_clipping_bound
            )
            noised_clipped_weights_diff = noise_utils.add_global_noise(
                self.model,
                clipped_weights_diff,
                self.random_state,
                self.local_sigma
                * self.local_clipping_bound
                * np.sqrt(self.n_silo_per_round),
                device=self.device,
            )
            return noised_clipped_weights_diff, len(train_loader)
        elif self.agg_strategy in ["SILO-LEVEL-DP"]:
            clipped_weights_diff = noise_utils.global_clip(
                self.model, weights_diff, self.local_clipping_bound
            )
            return clipped_weights_diff, len(train_loader)
        elif self.agg_strategy in [
            "ULDP-SGD",
            "ULDP-SGD-w",
            "ULDP-SGD-s",
            "ULDP-SGD-ws",
        ]:
            return noisy_avg_grads, len(self.train_loader)
        elif self.agg_strategy in [
            "ULDP-AVG",
            "ULDP-AVG-w",
            "ULDP-AVG-s",
            "ULDP-AVG-ws",
        ]:
            return noisy_avg_weights_diff, len(self.train_loader)
        elif self.agg_strategy in ["DEFAULT"]:
            return weights_diff, len(train_loader)
        else:
            raise NotImplementedError("Unknown aggregation strategy")

    def diff_weights(self, original_weights, udpated_weights):
        """Diff = Local - Global"""
        diff_weights = copy.deepcopy(udpated_weights)
        for key in diff_weights.keys():
            diff_weights[key] = udpated_weights[key] - original_weights[key]
        return diff_weights

    def test_local(self, round_idx=None):
        """
        Evaluate the model on the local dataset.
        """

        model = self.model
        model.to(self.device)
        model.eval()

        if self.agg_strategy in [
            "ULDP-SGD",
            "ULDP-SGD-w",
            "ULDP-AVG",
            "ULDP-AVG-w",
            "ULDP-SGD-s",
            "ULDP-AVG-s",
            "ULDP-AVG-ws",
            "ULDP-SGD-ws",
        ]:
            logger.debug("Skip local test as model is not trained locally")
            return

        if len(self.test_loader) == 0:
            logger.debug("Skip local test as dataset size is too small")
            return

        if self.dataset_name in ["heart_disease", "tcga_brca"]:
            with torch.no_grad():
                y_pred_final = []
                y_true_final = []
                n_total_data = 0
                test_loss = 0
                for idx, (x, y) in enumerate(self.test_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = model(x)
                    loss = self.criterion(y_pred, y)
                    test_loss += loss.item()
                    y_pred_final.append(y_pred.numpy())
                    y_true_final.append(y.numpy())
                    n_total_data += len(y)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            test_metric = self.metric(y_true_final, y_pred_final)
            logger.debug("|----- Local test result of round %d" % (round_idx))
            logger.debug(
                f"\t |----- Local Test/Acc: {test_metric} ({n_total_data}), Local Test/Loss: {test_loss}"
            )

        elif self.dataset_name in ["creditcard"]:
            from sklearn.metrics import roc_auc_score

            criterion = nn.CrossEntropyLoss().to(self.device)

            with torch.no_grad():
                n_total_data = 0
                test_loss = 0
                y_pred_final = []
                y_true_final = []
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y.long())
                    test_loss += loss.item()
                    y_pred = y_pred.argmax(dim=1)
                    y_pred_final.append(y_pred.numpy())
                    y_true_final.append(y.numpy())
                    n_total_data += len(y)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            test_metric = roc_auc_score(y_true_final, y_pred_final)
            logger.debug("|----- Local test result of round %d" % (round_idx))
            logger.debug(
                f"\t |----- Local Test/ROC_AUC: {test_metric} ({n_total_data}), Local Test/Loss: {test_loss}"
            )
        else:
            with torch.no_grad():
                n_total_data = 0
                test_loss = 0
                test_correct = 0
                for idx, (x, labels) in enumerate(self.test_loader):
                    x, labels = x.to(self.device), labels.to(self.device)
                    pred = model(x)
                    loss = self.criterion(pred, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(pred, 1)
                    test_correct += torch.sum(torch.eq(predicted, labels)).item()
                    n_total_data += len(labels)

            test_metric = test_correct / n_total_data
            logger.debug("|----- Local test result of round %d" % (round_idx))
            logger.debug(
                f"\t |----- Local Test/Acc: {test_metric} ({test_correct} / {n_total_data}), Local Test/Loss: {test_loss}"
            )
        self.results["local_test"].append((round_idx, test_metric, test_loss))
