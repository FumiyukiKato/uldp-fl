from collections import OrderedDict
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from opacus.accountants import RDPAccountant
from dataset import CREDITCARD, HEART_DISEASE, TCGA_BRCA
from method_group import (
    METHOD_DEFAULT,
    METHOD_GROUP_AGGREGATOR_PRIVACY_ACCOUNTING,
    METHOD_GROUP_ULDP_AVG,
    METHOD_GROUP_ULDP_SGD,
    METHOD_GROUP_ONLINE_OPTIMIZATION,
    METHOD_GROUP_ULDP_GROUPS,
    METHOD_GROUP_USER_LEVEL_DATA_LOADER,
    METHOD_GROUP_WITHIN_SILO_DP_ACCOUNTING,
    METHOD_PULDP_AVG,
    METHOD_PULDP_QC_TEST,
    METHOD_PULDP_QC_TRAIN,
    METHOD_RECORD_LEVEL_DP,
    METHOD_SILO_LEVEL_DP,
    METHOD_ULDP_NAIVE,
)

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
        C_u: Optional[Dict] = None,
        n_total_round: Optional[int] = None,
        hp_baseline: Optional[str] = None,
    ):
        self.base_seed = base_seed + silo_id + 1
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
        self.n_total_round = n_total_round
        self.local_delta = local_delta
        self.weight_decay = weight_decay
        self.client_optimizer = client_optimizer
        self.hp_baseline = hp_baseline

        self.results = {"local_test": [], "train_time": [], "epsilon": []}

        self.agg_strategy = agg_strategy
        if self.agg_strategy in METHOD_GROUP_WITHIN_SILO_DP_ACCOUNTING:
            assert client_optimizer == "sgd"
            from opacus import PrivacyEngine

            self.privacy_engine = PrivacyEngine(accountant="rdp")
            self.local_sigma = local_sigma
            self.local_clipping_bound = local_clipping_bound
            self.group_k = group_k
        elif self.agg_strategy in METHOD_GROUP_AGGREGATOR_PRIVACY_ACCOUNTING:
            self.local_sigma = local_sigma
            self.local_clipping_bound = local_clipping_bound
            self.C_u = C_u

        if self.dataset_name == HEART_DISEASE:
            from flamby_utils.heart_disease import (
                custom_loss,
                custom_optimizer,
                custom_metric,
            )

            self.criterion = custom_loss()
            self.optimizer = lambda model: custom_optimizer(
                self.model,
                self.local_learning_rate,
                self.client_optimizer,
                self.weight_decay,
            )
            self.metric = custom_metric()
        elif self.dataset_name == TCGA_BRCA:
            from flamby_utils.tcga_brca import (
                custom_loss,
                custom_optimizer,
                custom_metric,
            )

            self.criterion = custom_loss()
            self.optimizer = lambda model: custom_optimizer(
                model,
                self.local_learning_rate,
                self.client_optimizer,
                self.weight_decay,
            )
            self.metric = custom_metric()
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            if client_optimizer == "sgd":
                self.optimizer = lambda model: torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=local_learning_rate,
                )
            elif client_optimizer == "adam":
                self.optimizer = lambda model: torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=local_learning_rate,
                    weight_decay=weight_decay,
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
        if self.agg_strategy in METHOD_GROUP_USER_LEVEL_DATA_LOADER:
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

    def set_epsilon_groups(self, epsilon_groups: Dict[int, float]):
        self.epsilon_groups = epsilon_groups
        if self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
            self.accountant_dct: Dict[RDPAccountant] = {}
            for eps_u, _ in self.epsilon_groups.items():
                self.accountant_dct[eps_u] = RDPAccountant()

    def set_user_weights_with_optimization(
        self,
        user_weights: Dict[int, float],
        C_u_list: np.ndarray,
        user_weights_for_optimization: Dict[int, float],
        C_u_list_for_optimization: np.ndarray,
        stepped_user_weights_for_optimization: Dict[int, float],
        stepped_C_u_list_for_optimization: np.ndarray,
    ):
        self.user_weights = user_weights
        self.C_u_list = C_u_list
        self.user_weights_for_optimization = user_weights_for_optimization
        self.C_u_list_for_optimization = C_u_list_for_optimization
        self.stepped_user_weights_for_optimization = (
            stepped_user_weights_for_optimization
        )
        self.stepped_C_u_list_for_optimization = stepped_C_u_list_for_optimization

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
                if self.dataset_name not in [HEART_DISEASE, TCGA_BRCA]:
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
        self,
        global_round_index: int,
        loss_callback: Callable = lambda loss: None,
        off_train_loss_noise=None,
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
        optimizer = self.optimizer(model)
        criterion = self.criterion

        # Optimization step like CALCULATE GRADIENTS
        if self.agg_strategy in METHOD_GROUP_WITHIN_SILO_DP_ACCOUNTING:
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

        if self.agg_strategy in METHOD_GROUP_ULDP_SGD:
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
                    if self.dataset_name == CREDITCARD:
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
                avg_grads = noise_utils.torch_aggregation(grads_list)
            noisy_avg_grads = noise_utils.add_global_noise(
                model,
                avg_grads,
                self.random_state,
                self.local_sigma
                * self.local_clipping_bound
                / np.sqrt(self.n_silo_per_round),
                device=self.device,
            )

        elif self.agg_strategy in METHOD_GROUP_ULDP_AVG:
            # If suddenly becomes unstable, skip the update gradient (call step()) for now.
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
                if self.client_optimizer == "sgd":
                    optimizer_u = torch.optim.SGD(
                        filter(lambda p: p.requires_grad, model_u.parameters()),
                        lr=self.local_learning_rate,
                    )
                elif self.client_optimizer == "adam":
                    optimizer_u = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model_u.parameters()),
                        lr=self.local_learning_rate,
                        weight_decay=self.weight_decay,
                    )
                else:
                    raise ValueError("Unknown client optimizer")
                for epoch in range(self.local_epochs):
                    batch_loss = []
                    for x, labels in user_train_loader:
                        x, labels = x.to(self.device), labels.to(self.device)
                        optimizer_u.zero_grad()
                        log_probs = model_u(x)
                        if self.dataset_name == CREDITCARD:
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
                if noise_utils.check_nan_inf(model_u):
                    # If it includes Nan or Inf, then
                    pass
                else:
                    weights_diff = noise_utils.diff_weights(global_weights, weights)
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
                avg_weights_diff = noise_utils.torch_aggregation(weights_diff_list)
            noisy_avg_weights_diff = noise_utils.add_global_noise(
                model,
                avg_weights_diff,
                self.random_state,
                self.local_sigma
                * self.local_clipping_bound
                / np.sqrt(self.n_silo_per_round),
                device=self.device,
            )

        elif self.agg_strategy == METHOD_PULDP_AVG:
            # If suddenly becomes unstable, skip the update gradient (call step()) for now.
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
                if self.client_optimizer == "sgd":
                    optimizer_u = torch.optim.SGD(
                        filter(lambda p: p.requires_grad, model_u.parameters()),
                        lr=self.local_learning_rate,
                    )
                elif self.client_optimizer == "adam":
                    optimizer_u = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model_u.parameters()),
                        lr=self.local_learning_rate,
                        weight_decay=self.weight_decay,
                    )
                else:
                    raise ValueError("Unknown client optimizer")
                for epoch in range(self.local_epochs):
                    batch_loss = []
                    for x, labels in user_train_loader:
                        x, labels = x.to(self.device), labels.to(self.device)
                        optimizer_u.zero_grad()
                        log_probs = model_u(x)
                        if self.dataset_name == CREDITCARD:
                            labels = labels.long()
                        loss = criterion(log_probs, labels)
                        if loss_callback(loss):
                            continue
                        loss.backward()
                        optimizer_u.step()
                        batch_loss.append(loss.item())
                    if len(batch_loss) > 0:
                        logger.debug(
                            "Silo Id = {}\tEpoch: {}\tLoss: {:.6f}".format(
                                self.silo_id, epoch, sum(batch_loss) / len(batch_loss)
                            )
                        )
                weights = model_u.state_dict()
                weights_diff = noise_utils.diff_weights(global_weights, weights)
                clipped_weights_diff = noise_utils.global_clip(
                    model_u, weights_diff, self.C_u[user_id]
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
                avg_weights_diff = noise_utils.torch_aggregation(weights_diff_list)
            noisy_avg_weights_diff = noise_utils.add_global_noise(
                model,
                avg_weights_diff,
                self.random_state,
                self.local_sigma
                / np.sqrt(
                    self.n_silo_per_round
                ),  # this local_sigma is the standard deviation of normal dist itself
                device=self.device,
            )

        elif self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:

            def loss_callback(loss):
                if torch.isnan(loss):
                    logger.warn("loss is nan: skipping")
                    return True
                return False

            weights_diff_dct_per_epsilon_group = {}
            weights_diff_dct_per_epsilon_group_for_optimization = {}
            stepped_weights_diff_dct_per_epsilon_group_for_optimization = {}
            for eps_u, user_ids_per_eps_u in self.epsilon_groups.items():
                user_ids_per_eps_u_set = set(user_ids_per_eps_u)
                for ith, (user_weights, sensitivities, weights_diff_dct) in enumerate(
                    [
                        (
                            self.user_weights,
                            self.C_u_list,
                            weights_diff_dct_per_epsilon_group,
                        ),
                        (
                            self.user_weights_for_optimization,
                            self.C_u_list_for_optimization,
                            weights_diff_dct_per_epsilon_group_for_optimization,
                        ),
                        (
                            self.stepped_user_weights_for_optimization,
                            self.stepped_C_u_list_for_optimization,
                            stepped_weights_diff_dct_per_epsilon_group_for_optimization,
                        ),
                    ]
                ):
                    if ith > 0 and self.hp_baseline is not None:
                        break
                    weights_diff_list = []
                    for user_id, user_train_loader in self.user_level_data_loader:
                        # to compute model delta per epsilon group
                        if user_id not in user_ids_per_eps_u_set:
                            continue
                        if (  # not sampled users
                            user_weights[user_id] <= 0.0
                        ):  # for efficiency, if w is encrypted for DDP, it can't work
                            continue
                        model_u = copy.deepcopy(model)
                        if self.client_optimizer == "sgd":
                            optimizer_u = torch.optim.SGD(
                                filter(lambda p: p.requires_grad, model_u.parameters()),
                                lr=self.local_learning_rate,
                            )
                        elif self.client_optimizer == "adam":
                            optimizer_u = torch.optim.Adam(
                                filter(lambda p: p.requires_grad, model_u.parameters()),
                                lr=self.local_learning_rate,
                                weight_decay=self.weight_decay,
                            )
                        else:
                            raise ValueError("Unknown client optimizer")
                        for epoch in range(self.local_epochs):
                            batch_loss = []
                            for x, labels in user_train_loader:
                                x, labels = x.to(self.device), labels.to(self.device)
                                optimizer_u.zero_grad()
                                log_probs = model_u(x)
                                if self.dataset_name == CREDITCARD:
                                    labels = labels.long()
                                loss = criterion(log_probs, labels)
                                if loss_callback(loss):
                                    continue
                                loss.backward()
                                optimizer_u.step()
                                batch_loss.append(loss.item())
                            if len(batch_loss) > 0:
                                logger.debug(
                                    "Silo Id = {}\tEpoch: {}\tLoss: {:.6f}".format(
                                        self.silo_id,
                                        epoch,
                                        sum(batch_loss) / len(batch_loss),
                                    )
                                )
                        weights = model_u.state_dict()
                        weights_diff = noise_utils.diff_weights(global_weights, weights)
                        clipped_weights_diff = noise_utils.global_clip(
                            model_u, weights_diff, sensitivities[user_id]
                        )
                        weighted_clipped_weights_diff = noise_utils.multiple_weights(
                            model_u, clipped_weights_diff, user_weights[user_id]
                        )
                        weights_diff_list.append(weighted_clipped_weights_diff)

                    # average weights diff in the same epsilon group
                    if len(weights_diff_list) <= 0:
                        avg_weights_diff = OrderedDict()
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                avg_weights_diff[name] = torch.zeros_like(param.data)
                    else:
                        avg_weights_diff = noise_utils.torch_aggregation(
                            weights_diff_list
                        )
                    weights_diff_dct[eps_u] = avg_weights_diff

            # compute delta(q_u, C_u)
            default_avg_weights_diff = noise_utils.torch_aggregation(
                list(weights_diff_dct_per_epsilon_group.values())
            )

            default_noisy_avg_weights_diff = noise_utils.add_global_noise(
                model,
                default_avg_weights_diff,
                self.random_state,
                self.local_sigma
                / np.sqrt(
                    self.n_silo_per_round
                ),  # this local_sigma is the standard deviation of normal dist itself
                device=self.device,
            )
            DEFAULT_NAME = "default"
            noisy_avg_weights_diff_dct = {DEFAULT_NAME: default_noisy_avg_weights_diff}

            if self.hp_baseline:
                return noisy_avg_weights_diff_dct

            # For finite difference method, compute delta(q_u_i, C_u_i)
            for eps_u in self.epsilon_groups.keys():
                # Need to compute delta(q_u_i, C_u_i) for each eps_i
                # They need to be noised for DP in distributed manner
                if self.agg_strategy == METHOD_PULDP_QC_TEST:
                    original_avg_weights_diff = noise_utils.torch_aggregation(
                        [weights_diff_dct_per_epsilon_group_for_optimization[eps_u]]
                    )
                    if off_train_loss_noise:
                        noisy_original_avg_weights_diff = noise_utils.add_global_noise(
                            model,
                            original_avg_weights_diff,
                            self.random_state,
                            0.0000000000000001,
                            device=self.device,
                        )
                    else:
                        noisy_original_avg_weights_diff = noise_utils.add_global_noise(
                            model,
                            original_avg_weights_diff,
                            self.random_state,
                            self.local_sigma
                            / np.sqrt(
                                self.n_silo_per_round
                            ),  # this local_sigma is the standard deviation of normal dist itself
                            device=self.device,
                        )
                    stepped_avg_weights_diff = noise_utils.torch_aggregation(
                        [
                            stepped_weights_diff_dct_per_epsilon_group_for_optimization[
                                eps_u
                            ]
                        ]
                    )
                    if off_train_loss_noise:
                        noisy_stepped_avg_weights_diff = noise_utils.add_global_noise(
                            model,
                            stepped_avg_weights_diff,
                            self.random_state,
                            0.0000000000000001,
                            device=self.device,
                        )
                    else:
                        noisy_stepped_avg_weights_diff = noise_utils.add_global_noise(
                            model,
                            stepped_avg_weights_diff,
                            self.random_state,
                            self.local_sigma
                            / np.sqrt(
                                self.n_silo_per_round
                            ),  # this local_sigma is the standard deviation of normal dist itself
                            device=self.device,
                        )

                    noisy_avg_weights_diff_dct[eps_u] = (
                        noisy_original_avg_weights_diff,
                        noisy_stepped_avg_weights_diff,
                    )

                elif self.agg_strategy == METHOD_PULDP_QC_TRAIN:
                    # doesn't need noise
                    original_avg_weights_diff = noise_utils.torch_aggregation(
                        [weights_diff_dct_per_epsilon_group_for_optimization[eps_u]]
                    )
                    stepped_avg_weights_diff = noise_utils.torch_aggregation(
                        [
                            stepped_weights_diff_dct_per_epsilon_group_for_optimization[
                                eps_u
                            ]
                        ]
                    )

                    noisy_avg_weights_diff_dct[eps_u] = (
                        original_avg_weights_diff,
                        stepped_avg_weights_diff,
                    )
                else:
                    raise NotImplementedError("Unknown aggregation strategy")

        else:
            for epoch in range(self.local_epochs):
                batch_loss = []
                for x, labels in train_loader:
                    if len(x) == 0:  # this is possible in poisson sampling in DP-SGD
                        continue
                    x, labels = x.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    log_probs = model(x)
                    if self.dataset_name == CREDITCARD:
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
                logger.debug(
                    "Silo Id = {}\tEpoch: {}\tLoss: {:.6f}".format(
                        self.silo_id, epoch, sum(batch_loss) / len(batch_loss)
                    )
                )

            weights = self.get_model_params()
            weights_diff = noise_utils.diff_weights(global_weights, weights)

        train_time = time.time() - tick
        logger.debug("Train/Time : %s", train_time)
        self.results["train_time"].append((global_round_index, train_time))

        # Post-process step
        if self.agg_strategy == METHOD_RECORD_LEVEL_DP:
            model.remove_hooks()
            eps = self.privacy_engine.get_epsilon(delta=self.local_delta)
            logger.debug(
                "Silo Id = {}\tEpsilon: {:.6f} (delta: {:6f})".format(
                    self.silo_id, eps, self.local_delta
                )
            )
            self.results["epsilon"].append((global_round_index, eps))
            return weights_diff, len(train_loader)
        elif self.agg_strategy in METHOD_GROUP_ULDP_GROUPS:
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
        elif self.agg_strategy == METHOD_ULDP_NAIVE:
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
        elif self.agg_strategy == METHOD_SILO_LEVEL_DP:
            clipped_weights_diff = noise_utils.global_clip(
                self.model, weights_diff, self.local_clipping_bound
            )
            return clipped_weights_diff, len(train_loader)
        elif self.agg_strategy in METHOD_GROUP_ULDP_SGD:
            return noisy_avg_grads, len(self.train_loader)
        elif self.agg_strategy in METHOD_GROUP_ULDP_AVG:
            return noisy_avg_weights_diff, len(self.train_loader)
        elif self.agg_strategy == METHOD_PULDP_AVG:
            return noisy_avg_weights_diff, len(self.train_loader)
        elif self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
            return noisy_avg_weights_diff_dct
        elif self.agg_strategy == METHOD_DEFAULT:
            return weights_diff, len(train_loader)
        else:
            raise NotImplementedError("Unknown aggregation strategy")

    def test_local(self, round_idx=None):
        """
        Evaluate the model on the local dataset.
        """

        model = self.model
        model.to(self.device)
        model.eval()

        if self.agg_strategy in (METHOD_GROUP_ULDP_SGD | METHOD_GROUP_ULDP_AVG):
            logger.debug("Skip local test as model is not trained locally")
            return

        if len(self.test_loader) == 0:
            logger.debug("Skip local test as dataset size is too small")
            return

        if self.dataset_name in [HEART_DISEASE, TCGA_BRCA]:
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

        elif self.dataset_name == CREDITCARD:
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

    def train_loss(
        self,
        round_idx=None,
        model: Optional[nn.Module] = None,
    ) -> Tuple[float, float]:
        if model is None:
            model = self.model
        model.to(self.device)
        model.eval()

        user_level_total_loss = []
        user_level_metrics = []
        if self.dataset_name == TCGA_BRCA:
            logger.warning("TCGA_BRCA does not support train_loss")
            return 0, 0

        for user_id, user_train_loader in self.user_level_data_loader:
            if self.dataset_name in [HEART_DISEASE]:
                with torch.no_grad():
                    y_pred_final = []
                    y_true_final = []
                    n_total_data = 0
                    train_loss = 0
                    for x, y in user_train_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        y_pred = model(x)
                        loss = self.criterion(y_pred, y)
                        train_loss += loss.item()
                        y_pred_final.append(y_pred.numpy())
                        y_true_final.append(y.numpy())
                        n_total_data += len(y)

                y_true_final = np.concatenate(y_true_final)
                y_pred_final = np.concatenate(y_pred_final)
                train_metric = self.metric(y_true_final, y_pred_final)
                logger.debug(
                    f"|----- Local test result of round {round_idx}, user {user_id}"
                )
                logger.debug(
                    f"\t |----- Local Train/Acc: {train_metric} ({n_total_data}), Local Train/Loss: {train_loss}"
                )

            elif self.dataset_name == CREDITCARD:
                from sklearn.metrics import roc_auc_score

                criterion = nn.CrossEntropyLoss().to(self.device)

                with torch.no_grad():
                    n_total_data = 0
                    train_loss = 0
                    y_pred_final = []
                    y_true_final = []
                    for x, y in user_train_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        y_pred = model(x)
                        loss = criterion(y_pred, y.long())
                        train_loss += loss.item()
                        y_pred = y_pred.argmax(dim=1)
                        y_pred_final.append(y_pred.numpy())
                        y_true_final.append(y.numpy())
                        n_total_data += len(y)

                y_true_final = np.concatenate(y_true_final)
                y_pred_final = np.concatenate(y_pred_final)
                train_metric = roc_auc_score(y_true_final, y_pred_final)
                logger.debug(
                    f"|----- Local test result of round {round_idx}, user {user_id}"
                )
                logger.debug(
                    f"\t |----- Local Test/ROC_AUC: {train_metric} ({n_total_data}), Local Test/Loss: {train_loss}"
                )
            else:
                with torch.no_grad():
                    n_total_data = 0
                    train_loss = 0
                    test_correct = 0
                    for x, labels in user_train_loader:
                        x, labels = x.to(self.device), labels.to(self.device)
                        pred = model(x)
                        loss = self.criterion(pred, labels)
                        train_loss += loss.item()
                        _, predicted = torch.max(pred, 1)
                        test_correct += torch.sum(torch.eq(predicted, labels)).item()
                        n_total_data += len(labels)

                train_metric = test_correct / n_total_data
                logger.debug(
                    f"|----- Local test result of round {round_idx}, user {user_id}"
                )
                logger.debug(
                    f"\t |----- Local Test/Acc: {train_metric} ({test_correct} / {n_total_data}), Local Test/Loss: {train_loss}"
                )

            user_level_total_loss.append(train_loss / n_total_data)
            user_level_metrics.append(train_metric)

        return sum(user_level_total_loss), sum(user_level_metrics)

    def dp_train_loss(
        self,
        eps_u: float,
        user_weights: Dict[int, float],
        round_idx=None,
        model: Optional[nn.Module] = None,
        sampling_rate_q: Optional[float] = None,
        current_round: Optional[int] = None,
        off_train_loss_noise: Optional[bool] = False,
    ) -> Tuple[Tuple[float, float], float]:
        # metric is approximated metric instead of the real train loss
        # it needs to be bounded by 1
        # privacy accounting is done in aggregator.py consume_dp_for_train_loss_metric()

        if model is None:
            model = self.model
        model.to(self.device)
        model.eval()

        if self.dataset_name == TCGA_BRCA:
            logger.warning(
                "TCGA_BRCA does not currently support train_loss because cox-loss is not well-compatible for user-level setting"
            )
            return 0, 0

        loss_list = []
        metric_list = []
        raw_metric_list = []

        user_ids_per_eps_u_set = set(self.epsilon_groups[eps_u])

        for user_id, user_train_loader in self.user_level_data_loader:
            # to compute model delta per epsilon group
            if user_id not in user_ids_per_eps_u_set:
                continue
            if (  # not sampled users
                user_weights[user_id] <= 0.0
            ):  # for efficiency, if w is encrypted for DDP, it can't work
                continue

            if self.dataset_name in [HEART_DISEASE]:
                with torch.no_grad():
                    y_pred_final = []
                    y_true_final = []
                    n_total_data = 0
                    train_loss = 0
                    for x, y in user_train_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        y_pred = model(x)
                        loss = self.criterion(y_pred, y)
                        train_loss += loss.item()
                        y_pred_final.append(y_pred.numpy())
                        y_true_final.append(y.numpy())
                        n_total_data += len(y)

                y_true_final = np.concatenate(y_true_final)
                y_pred_final = np.concatenate(y_pred_final)
                train_metric = self.metric(y_true_final, y_pred_final)
                logger.debug(
                    f"|----- Local test result of round {round_idx}, user {user_id})"
                )
                logger.debug(
                    f"\t |----- Local Train/Acc: {train_metric} ({n_total_data}), Local Train/Loss: {train_loss}"
                )

            elif self.dataset_name == CREDITCARD:
                from sklearn.metrics import roc_auc_score

                criterion = nn.CrossEntropyLoss().to(self.device)

                with torch.no_grad():
                    n_total_data = 0
                    train_loss = 0
                    y_pred_final = []
                    y_true_final = []
                    for x, y in user_train_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        y_pred = model(x)
                        loss = criterion(y_pred, y.long())
                        train_loss += loss.item()
                        y_pred = y_pred.argmax(dim=1)
                        y_pred_final.append(y_pred.numpy())
                        y_true_final.append(y.numpy())
                        n_total_data += len(y)

                y_true_final = np.concatenate(y_true_final)
                y_pred_final = np.concatenate(y_pred_final)
                # ROC AUC is hard to use in case the label is extremely imbalanced
                # train_metric = roc_auc_score(y_true_final, y_pred_final)
                # Accuracy
                # train_metric = np.sum(y_true_final == y_pred_final) / len(y_true_final)
                if np.any(y_true_final == 1) and np.any(
                    y_true_final == 0
                ):  # ポジティブクラスが存在する場合
                    train_metric = roc_auc_score(
                        y_true_final, y_pred_final
                    )  # またはrecall_score, f1_scoreなど
                else:
                    train_metric = np.sum(y_true_final == y_pred_final) / len(
                        y_true_final
                    )
                    train_metric = train_metric * 0.1  # lower the importance

                logger.debug(
                    f"|----- Local test result of round {round_idx}, user {user_id}"
                )
                logger.debug(
                    f"\t |----- Local Test/Accuracy: {train_metric} ({n_total_data}), Local Test/Loss: {train_loss}"
                )

            else:
                with torch.no_grad():
                    n_total_data = 0
                    train_loss = 0
                    test_correct = 0
                    for x, labels in user_train_loader:
                        x, labels = x.to(self.device), labels.to(self.device)
                        pred = model(x)
                        loss = self.criterion(pred, labels)
                        train_loss += loss.item()
                        _, predicted = torch.max(pred, 1)
                        test_correct += torch.sum(torch.eq(predicted, labels)).item()
                        n_total_data += len(labels)

                train_metric = test_correct / n_total_data
                logger.debug(
                    f"|----- Local test result of round {round_idx}, user {user_id}"
                )
                logger.debug(
                    f"\t |----- Local Test/Acc: {train_metric} ({test_correct} / {n_total_data}), Local Test/Loss: {train_loss}"
                )

            # the metric is bounded by C_u per user
            user_level_shrunk_metric = train_metric * user_weights[user_id]
            loss_list.append(train_loss / n_total_data)
            metric_list.append(user_level_shrunk_metric)
            raw_metric_list.append(train_metric)

        logger.debug("metric_list: (", len(metric_list), ")", metric_list)
        noise_multiplier, _ = noise_utils.get_noise_multiplier_with_history(
            sampling_rate_q,
            self.local_delta,
            epsilon_u=eps_u,
            total_round=self.n_total_round * 3,
            current_round=current_round,
            current_accountant=self.accountant_dct[eps_u],
        )
        if off_train_loss_noise:
            noise = noise_utils.single_gaussian_noise(
                random_state=self.random_state, std_dev=0.0000000000001
            )
        else:
            noise = noise_utils.single_gaussian_noise(
                random_state=self.random_state,
                std_dev=noise_multiplier / np.sqrt(self.n_silo_per_round),
                # sensitivity is same for all users in the same epsilon group, and add distributed noise here
            )
        logger.debug("noise:", noise)
        final_shrunk_noisy_metric = np.sum(metric_list) + noise
        final_shrunk_noisy_metric /= sampling_rate_q

        return final_shrunk_noisy_metric, noise_multiplier

    def consume_dp_for_model_optimization(self, q_u_list):
        # (Consume privacy) Update local accountants for model aggregation for each user groups
        for eps_u, eps_user_ids in self.epsilon_groups.items():
            q_u = q_u_list[eps_user_ids[0]]
            C_u = self.C_u_list[eps_user_ids[0]]
            self.accountant_dct[eps_u].step(
                noise_multiplier=self.local_sigma / C_u,
                sample_rate=q_u,
            )

    def consume_dp_for_stepped_model_optimization(self, stepped_q_u_list):
        # (Consume privacy) Update local accountants for model aggregation for each user groups
        for eps_u, eps_user_ids in self.epsilon_groups.items():
            q_u = stepped_q_u_list[eps_user_ids[0]]
            C_u = self.stepped_C_u_list_for_optimization[eps_user_ids[0]]
            self.accountant_dct[eps_u].step(
                noise_multiplier=self.local_sigma / C_u,
                sample_rate=q_u,
            )

    def compute_train_loss_for_online_optimization_and_consume_dp_for_train_loss_metric(
        self,
        round_idx: int,
        q_u_list: List,
        stepped_q_u_list: List,
        local_updated_weights_dct: Dict,
        off_train_loss_noise: bool = False,
    ) -> Dict[float, float]:
        # Calculate the difference of the train loss (or approximated train loss like user level accuracy) between the original and the updated sampling rate
        # At the same time, consume privacy for differentially private training loss (or user level accuracy)
        diff_dct = {}

        for eps_u, eps_user_ids in self.epsilon_groups.items():
            q_u = q_u_list[eps_user_ids[0]]

            original_weights_diff, stepped_weights_diff = local_updated_weights_dct[
                eps_u
            ]

            # Calculate train loss with original sampling rate for approximated HP gradients
            model = copy.deepcopy(self.model)
            averaged_param_diff = noise_utils.torch_aggregation(
                [original_weights_diff],
                np.ceil(len(eps_user_ids) * self.n_silo_per_round * q_u),
            )
            noise_utils.update_global_weights_from_diff(
                averaged_param_diff,
                model,
                learning_rate=1.0,
            )

            (
                original_user_level_metric,
                local_noise_multiplier,
            ) = self.dp_train_loss(
                eps_u=eps_u,
                user_weights=self.user_weights_for_optimization,
                round_idx=round_idx,
                model=model,
                sampling_rate_q=q_u,
                current_round=round_idx * 3 + 1,
                off_train_loss_noise=off_train_loss_noise,
            )
            # (Consume privacy)
            self.accountant_dct[eps_u].step(
                noise_multiplier=local_noise_multiplier,
                sample_rate=q_u,
            )
            logger.debug(
                "Original sampling_rate_q = {}, metric = {}".format(
                    q_u, original_user_level_metric
                )
            )

            # Calculate train loss with updated (stepped) sampling rate
            model = copy.deepcopy(self.model)
            stepped_q_u = stepped_q_u_list[eps_user_ids[0]]
            averaged_param_diff = noise_utils.torch_aggregation(
                [stepped_weights_diff],
                np.ceil(len(eps_user_ids) * self.n_silo_per_round * stepped_q_u),
            )
            noise_utils.update_global_weights_from_diff(
                averaged_param_diff,
                model,
                learning_rate=1.0,
            )

            (
                stepped_user_level_metric,
                stepped_local_noise_multiplier,
            ) = self.dp_train_loss(
                eps_u=eps_u,
                user_weights=self.stepped_user_weights_for_optimization,
                round_idx=round_idx,
                model=model,
                sampling_rate_q=stepped_q_u,
                current_round=round_idx * 3 + 2,
                off_train_loss_noise=off_train_loss_noise,
            )
            # (Consume privacy)
            self.accountant_dct[eps_u].step(
                noise_multiplier=stepped_local_noise_multiplier,
                sample_rate=stepped_q_u,
            )
            logger.debug(
                "Stepped sampling_rate_q = {}, metric = {}".format(
                    stepped_q_u, stepped_user_level_metric
                )
            )

            # diff < 0 means that the model is improved
            diff = original_user_level_metric - stepped_user_level_metric

            diff_dct[eps_u] = diff
            logger.debug("eps_u = {}, diff = {}".format(eps_u, diff))

        return diff_dct
