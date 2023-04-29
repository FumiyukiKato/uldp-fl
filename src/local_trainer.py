import time
from typing import List, Tuple
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader

from mylogger import logger


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
        client_optimizer: str = "sgd",
        learning_rate: float = 0.001,
        local_batch_size: int = 1,
        weight_decay: float = 0.001,
        epochs: int = 5,
        local_delta: float = None,
        local_sigma: float = None,
        local_clipping_bound: float = None,
    ):
        self.random_state = np.random.RandomState(seed=base_seed + silo_id + 1)
        self.model: nn.Module = model
        self.silo_id = silo_id
        self.device = device
        self.epochs = epochs

        self.results = {"local_test": [], "train_time": []}

        self.agg_strategy = agg_strategy
        if self.agg_strategy in ["RECORD-LEVEL-DP", "GROUP-DP"]:
            assert client_optimizer == "sgd"
            from opacus import PrivacyEngine

            self.privacy_engine = PrivacyEngine(accountant="rdp")
            self.local_delta = local_delta
            self.local_sigma = local_sigma
            self.local_clipping_bound = local_clipping_bound
            self.results["epsilon"] = []

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_loader = DataLoader(
            local_train_dataset, batch_size=local_batch_size, shuffle=True
        )
        self.test_loader = DataLoader(local_test_dataset)

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

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, global_round_index: int):
        """
        Train the model on the local dataset.
        """
        tick = time.time()

        model = self.model
        model.to(self.device)
        model.train()

        torch.manual_seed(self.get_torch_manual_seed())

        train_loader = self.train_loader
        optimizer = self.optimizer
        criterion = self.criterion

        if self.agg_strategy in ["RECORD-LEVEL-DP"]:  # TODO: "GROUP-DP"
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

        if self.agg_strategy in ["RECORD-LEVEL-DP"]:
            eps = self.privacy_engine.get_epsilon(delta=self.local_delta)
            logger.info(
                "Silo Id = {}\tEpsilon: {:.6f} (delta: {:6f})".format(
                    self.silo_id, eps, self.local_delta
                )
            )
            self.results["epsilon"].append((global_round_index, eps))

        train_time = time.time() - tick
        logger.info("Train/Time : %s", train_time)
        self.results["train_time"].append((global_round_index, train_time))
        weights = self.get_model_params()

        if self.agg_strategy in ["RECORD-LEVEL-DP"]:
            model.remove_hooks()

        return weights, len(train_loader)

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

        return metrics
