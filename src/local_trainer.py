import time
from typing import Dict, List, Optional, Tuple
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
        learning_rate: float = 0.001,
        local_batch_size: int = 1,
        weight_decay: float = 0.001,
        epochs: int = 5,
        local_delta: Optional[float] = None,
        local_sigma: Optional[float] = None,
        local_clipping_bound: Optional[float] = None,
        group_k: Optional[int] = None,
    ):
        self.random_state = np.random.RandomState(seed=base_seed + silo_id + 1)
        self.model: nn.Module = model
        self.silo_id = silo_id
        self.device = device
        self.epochs = epochs
        self.local_batch_size = local_batch_size

        self.results = {"local_test": [], "train_time": [], "epsilon": []}

        self.agg_strategy = agg_strategy
        if self.agg_strategy in ["RECORD-LEVEL-DP", "ULDP-GROUP"]:
            assert client_optimizer == "sgd"
            from opacus import PrivacyEngine

            self.privacy_engine = PrivacyEngine(accountant="rdp")
            self.local_delta = local_delta
            self.local_sigma = local_sigma
            self.local_clipping_bound = local_clipping_bound
            self.group_k = group_k
        elif self.agg_strategy in ["ULDP-NAIVE", "SILO-LEVEL-DP"]:
            self.local_sigma = local_sigma
            self.local_clipping_bound = local_clipping_bound

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_loader = DataLoader(
            local_train_dataset, batch_size=local_batch_size, shuffle=True
        )
        self.test_loader = DataLoader(local_test_dataset)
        self.user_user_histogram = user_histogram
        self.user_ids_of_local_train_dataset = user_ids_of_local_train_dataset

        if client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate,
            )
        elif client_optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay,
                amsgrad=True,
            )
        else:
            raise ValueError("Unknown client optimizer")

    def get_results(self):
        return self.results

    def get_comm_results(self):
        return {}

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def get_torch_manual_seed(self):
        return self.random_state.randint(9223372036854775807)

    def get_latest_epsilon(self):
        if len(self.results["epsilon"]) == 0:
            return 0.0
        return self.results["epsilon"][-1][1]

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def bound_user_contributions(self, bounded_user_histogram):
        new_local_train_dataset = []
        new_local_test_dataset = self.test_loader.dataset

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
            else:
                remove_counter += 1
                new_local_test_dataset.append(data)
        logger.info("{} data is removed from training dataset".format(remove_counter))

        self.train_loader = DataLoader(
            new_local_train_dataset, batch_size=self.local_batch_size, shuffle=True
        )
        self.test_loader = DataLoader(new_local_test_dataset)

    def train(self, global_round_index: int):
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

        if self.agg_strategy in ["RECORD-LEVEL-DP", "ULDP-GROUP"]:
            noise_generator = torch.Generator().manual_seed(
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

        for epoch in range(self.epochs):
            batch_loss = []

            for idx, (x, labels) in enumerate(train_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            logger.info(
                "Silo Id = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.silo_id, epoch, sum(batch_loss) / len(batch_loss)
                )
            )

        train_time = time.time() - tick
        logger.info("Train/Time : %s", train_time)
        self.results["train_time"].append((global_round_index, train_time))
        weights = self.get_model_params()
        weights_dff = self.diff_weights(global_weights, weights)

        if self.agg_strategy in ["RECORD-LEVEL-DP"]:
            model.remove_hooks()
            eps = self.privacy_engine.get_epsilon(delta=self.local_delta)
            logger.info(
                "Silo Id = {}\tEpsilon: {:.6f} (delta: {:6f})".format(
                    self.silo_id, eps, self.local_delta
                )
            )
            self.results["epsilon"].append((global_round_index, eps))
            return weights_dff, len(train_loader)
        elif self.agg_strategy in ["ULDP-GROUP"]:
            model.remove_hooks()
            group_eps, opt_alpha = noise_utils.get_group_privacy_spent(
                group_k=self.group_k,
                accountant_history=self.privacy_engine.accountant.history,
                delta=self.local_delta,
            )
            logger.info(
                "Silo Id = {}\t (Group-Privacy) Epsilon: {:.6f} (delta: {:6f})".format(
                    self.silo_id, group_eps, self.local_delta
                )
            )
            self.results["epsilon"].append((global_round_index, group_eps))
            return weights_dff, len(train_loader)
        elif self.agg_strategy in ["ULDP-NAIVE"]:
            clipped_weights_dff = noise_utils.global_clip(
                weights_dff, self.local_clipping_bound
            )
            noised_clipped_weights_dff = noise_utils.add_global_noise(
                clipped_weights_dff,
                self.random_state,
                self.local_sigma * self.local_clipping_bound,
            )
            return noised_clipped_weights_dff, len(train_loader)
        elif self.agg_strategy in ["SILO-LEVEL-DP"]:
            clipped_weights_dff = noise_utils.global_clip(
                weights_dff, self.local_clipping_bound
            )
            return clipped_weights_dff, len(train_loader)
        elif self.agg_strategy in ["DEFAULT"]:
            return weights_dff, len(train_loader)
        else:
            raise NotImplementedError("Unknown aggregation strategy")

    def diff_weights(self, original_weights, udpated_weights):
        """Diff = Local - Global"""
        diff_weights = original_weights
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

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        if len(self.test_loader) == 0:
            logger.info("Skip local test as dataset size is too small")
            return

        with torch.no_grad():
            for idx, (x, labels) in enumerate(self.test_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                pred = model(x)

                loss = self.criterion(pred, labels)
                metrics["test_loss"] += loss.item()

                _, predicted = torch.max(pred, 1)
                metrics["test_correct"] += torch.sum(torch.eq(predicted, labels)).item()

                metrics["test_total"] += len(labels)

        test_acc = metrics["test_correct"] / metrics["test_total"]
        test_loss = metrics["test_loss"]
        logger.info("|----- Local test result of round %d" % (round_idx))
        logger.info(
            f"\t |----- Local Test/Acc: {test_acc} ({metrics['test_correct']} / {metrics['test_total']}), Local Test/Loss: {test_loss}"
        )
        self.results["local_test"].append((round_idx, test_acc, test_loss))
