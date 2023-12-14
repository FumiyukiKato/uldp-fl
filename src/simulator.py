from typing import Callable, List, Optional, Tuple, Dict
import torch
import numpy as np
import copy

from aggregator import Aggregator
from dataset import TCGA_BRCA
from method_group import (
    METHOD_GROUP_NEED_USER_RECORD,
    METHOD_GROUP_NO_SAMPLING,
    METHOD_GROUP_ONLINE_OPTIMIZATION,
    METHOD_GROUP_SAMPLING,
    METHOD_GROUP_ULDP_GROUPS,
    METHOD_GROUP_WEIGHTS,
    METHOD_PULDP_AVG,
    METHOD_PULDP_AVG_ONLINE,
    METHOD_PULDP_AVG_ONLINE_TRAIN,
)
from coordinator import Coordinator
from local_trainer import ClassificationTrainer
from mylogger import logger


class FLSimulator:
    """
    Federated learning simulator on memory.
    """

    def __init__(
        self,
        seed: int,
        model: torch.nn.Module,
        train_dataset: List[Tuple[torch.Tensor, int]],
        test_dataset: List[Tuple[torch.Tensor, int]],
        local_dataset_per_silos: Dict[
            int,
            Tuple[
                List[Tuple[torch.Tensor, int]],
                List[Tuple[torch.Tensor, int]],
                Dict[int, int],
                List[int],
            ],
        ],
        n_silos: int,
        n_users: int,
        device: str,
        n_total_round: int,
        n_silo_per_round: int,
        local_learning_rate: float,
        global_learning_rate: float,
        local_batch_size: int,
        weight_decay: float,
        client_optimizer: str,
        local_epochs: int,
        agg_strategy: str,
        clipping_bound: Optional[float] = None,
        sigma: Optional[float] = None,
        delta: Optional[float] = None,
        group_k: Optional[int] = None,
        dataset_name: str = None,
        sampling_rate_q: Optional[float] = None,
        C_u: Optional[Dict] = None,
        q_u: Optional[Dict] = None,
        epsilon_u: Optional[Dict] = None,
        group_thresholds: Optional[List] = None,
        q_step_size: Optional[float] = None,
        validation_ratio: Optional[float] = 0.0,
        with_momentum: Optional[bool] = False,
        train_loss_dp: Optional[bool] = False,
        momentum_weight: Optional[float] = None,
        sigma_for_online_optimization: Optional[float] = None,
        total_dp_eps_for_online_optimization: Optional[bool] = None,
    ):
        self.n_total_round = n_total_round
        self.round_idx = 0
        model.to(device)
        self.agg_strategy = agg_strategy
        self.dataset_name = dataset_name
        self.sampling_rate_q = sampling_rate_q
        self.validation_ratio = validation_ratio
        self.coordinator = Coordinator(
            base_seed=seed,
            n_silos=n_silos,
            n_users=n_users,
            group_k=group_k,
            sampling_rate_q=sampling_rate_q,
            agg_strategy=agg_strategy,
            q_u=q_u,
            epsilon_u=epsilon_u,
            group_thresholds=group_thresholds,
            delta=delta,
            sigma=sigma,
            n_total_round=n_total_round,
            q_step_size=q_step_size,
            train_loss_dp=train_loss_dp,
            total_dp_eps_for_online_optimization=total_dp_eps_for_online_optimization,
        )

        self.aggregator = Aggregator(
            model=copy.deepcopy(model),
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_users=n_users,
            n_silos=n_silos,
            n_silo_per_round=n_silo_per_round,
            device=device,
            base_seed=seed,
            strategy=agg_strategy,
            clipping_bound=clipping_bound,
            sigma=sigma,
            delta=delta,
            global_learning_rate=global_learning_rate,
            dataset_name=dataset_name,
            sampling_rate_q=sampling_rate_q,
            validation_ratio=validation_ratio,
            sigma_for_online_optimization=sigma_for_online_optimization,
            total_dp_eps_for_online_optimization=total_dp_eps_for_online_optimization,
            n_total_round=n_total_round,
        )

        if self.agg_strategy == METHOD_PULDP_AVG:
            self.aggregator.sampling_rate_q = np.mean(list(q_u.values()))
        if self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
            self.with_momentum = with_momentum
            self.train_loss_dp = train_loss_dp
            if momentum_weight is None:
                self.momentum_weight = q_step_size
            else:
                self.momentum_weight = momentum_weight
            self.aggregator.set_epsilon_groups(self.coordinator.epsilon_groups)
            if validation_ratio <= 0.0:
                raise ValueError(
                    "validation ratio must be greater than 0.0 for online optimization with Test Loss"
                )

        self.local_trainer_per_silos: Dict[int, ClassificationTrainer] = {}
        for silo_id, (
            local_train_dataset,
            local_test_dataset,
            user_hist,
            user_ids,
        ) in local_dataset_per_silos.items():
            local_trainer = ClassificationTrainer(
                base_seed=seed,
                model=copy.deepcopy(model),
                silo_id=silo_id,
                agg_strategy=agg_strategy,
                device=device,
                local_train_dataset=local_train_dataset,
                local_test_dataset=local_test_dataset,
                user_histogram=user_hist,
                user_ids_of_local_train_dataset=user_ids,
                client_optimizer=client_optimizer,
                local_learning_rate=local_learning_rate,
                local_batch_size=local_batch_size,
                weight_decay=weight_decay,
                local_epochs=local_epochs,
                local_sigma=sigma,
                local_delta=delta,
                local_clipping_bound=clipping_bound,
                group_k=group_k,
                n_silo_per_round=n_silo_per_round,
                dataset_name=dataset_name,
                C_u=C_u,
                sigma_for_online_optimization=sigma_for_online_optimization,
                total_dp_eps_for_online_optimization=total_dp_eps_for_online_optimization,
                n_total_round=n_total_round,
            )
            self.local_trainer_per_silos[silo_id] = local_trainer
            if self.agg_strategy in METHOD_GROUP_NEED_USER_RECORD:
                self.coordinator.set_user_hist_by_silo_id(silo_id, user_hist)
            if self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
                local_trainer.set_epsilon_groups(self.coordinator.get_epsilon_groups())

        self.coordinator.is_ready()
        self.group_k = self.coordinator.group_k

    def run(self):
        logger.info("Start federated learning simulation")

        for local_trainer in self.local_trainer_per_silos.values():
            local_trainer.set_group_k(self.group_k)

        if self.agg_strategy in METHOD_GROUP_ULDP_GROUPS:
            if self.dataset_name == TCGA_BRCA:
                min_count = 2
            else:
                min_count = 1
            bounded_user_hist_per_silo = self.coordinator.build_user_bound_histograms(
                self.coordinator.original_user_hist_dct, min_count
            )

            for silo_id, bounded_user_hist in bounded_user_hist_per_silo.items():
                self.local_trainer_per_silos[silo_id].bound_user_contributions(
                    bounded_user_hist
                )
        elif self.agg_strategy in METHOD_GROUP_NO_SAMPLING:
            if self.agg_strategy in METHOD_GROUP_NO_SAMPLING.intersection(
                METHOD_GROUP_WEIGHTS
            ):
                user_weights_per_silo = self.coordinator.build_user_weights(
                    weighted=True
                )
            else:
                user_weights_per_silo = self.coordinator.build_user_weights(
                    weighted=False
                )
            for silo_id, user_weights in user_weights_per_silo.items():
                self.local_trainer_per_silos[silo_id].set_user_weights(user_weights)

        while self.round_idx < self.n_total_round:
            silo_id_list_in_this_round = self.aggregator.silo_selection()

            if self.agg_strategy in METHOD_GROUP_SAMPLING.difference(
                METHOD_GROUP_WEIGHTS
            ):
                user_weights_per_silo = self.coordinator.build_user_weights(
                    weighted=False, is_sample=True
                )
                for silo_id, user_weights in user_weights_per_silo.items():
                    self.local_trainer_per_silos[silo_id].set_user_weights(user_weights)

            elif self.agg_strategy in METHOD_GROUP_SAMPLING.intersection(
                METHOD_GROUP_WEIGHTS
            ):
                user_weights_per_silo = self.coordinator.build_user_weights(
                    weighted=True, is_sample=True
                )
                for silo_id, user_weights in user_weights_per_silo.items():
                    self.local_trainer_per_silos[silo_id].set_user_weights(user_weights)

            elif self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
                # compute stepped q_u and C_u to calculate finite difference
                (
                    user_weights_per_silo,
                    C_u_list,
                    user_weights_per_silo_for_optimization,
                    C_u_list_for_optimization,
                    stepped_user_weights_per_silo_for_optimization,
                    stepped_C_u_list_for_optimization,
                ) = self.coordinator.build_user_weights_with_online_optimization(
                    weighted=True,
                )
                for silo_id in user_weights_per_silo.keys():
                    self.local_trainer_per_silos[
                        silo_id
                    ].set_user_weights_with_optimization(
                        user_weights_per_silo[silo_id],
                        C_u_list,
                        user_weights_per_silo_for_optimization[silo_id],
                        C_u_list_for_optimization,
                        stepped_user_weights_per_silo_for_optimization[silo_id],
                        stepped_C_u_list_for_optimization,
                    )
                self.aggregator.sampling_rate_q = np.mean(self.coordinator.q_u_list)

            for silo_id in silo_id_list_in_this_round:
                logger.debug(
                    "============ TRAINING: SILO_ID = %d (ROUND %d) ============"
                    % (silo_id, self.round_idx)
                )
                local_trainer = self.local_trainer_per_silos[silo_id]
                local_trainer.set_model_params(
                    self.aggregator.get_global_model_params()
                )

                if self.agg_strategy == METHOD_PULDP_AVG_ONLINE:
                    local_updated_weights_dct = local_trainer.train(
                        self.round_idx,
                        loss_callback=build_loss_callback(),
                    )
                    self.aggregator.add_local_trained_result_with_static_optimization(
                        silo_id,
                        local_updated_weights_dct,
                        local_trainer.get_latest_epsilon(),
                    )
                elif self.agg_strategy == METHOD_PULDP_AVG_ONLINE_TRAIN:
                    local_updated_weights_dct = local_trainer.train(
                        self.round_idx,
                        loss_callback=build_loss_callback(),
                    )
                    local_loss_diff_dct = (
                        local_trainer.train_loss_for_online_optimization(
                            self.round_idx,
                            self.coordinator.q_u_list,
                            self.coordinator.stepped_q_u_list,
                            local_updated_weights_dct,
                            uldp=self.train_loss_dp,
                        )
                    )
                    self.aggregator.add_local_trained_result_with_online_optimization(
                        silo_id,
                        local_updated_weights_dct["default"],
                        local_trainer.get_latest_epsilon(),
                        local_loss_diff_dct,
                    )
                else:
                    local_updated_weights, _ = local_trainer.train(
                        self.round_idx,
                        loss_callback=build_loss_callback(),
                    )
                    self.aggregator.add_local_trained_result(
                        silo_id,
                        local_updated_weights,
                        local_trainer.get_latest_epsilon(),
                    )

            logger.debug(
                "============ AGGREGATION: ROUND %d ============" % (self.round_idx)
            )
            self.aggregator.aggregate(silo_id_list_in_this_round, self.round_idx)
            test_acc, test_loss = self.aggregator.test_global(self.round_idx)
            if self.validation_ratio > 0.0:
                self.aggregator.test_global(self.round_idx, is_validation=True)
            if self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
                self.aggregator.consume_dp_for_model_optimization(
                    q_u_list=self.coordinator.q_u_list,
                    C_u_list=self.coordinator.C_u_list,
                )
                loss_diff_dct = self.aggregator.compute_loss_diff(
                    silo_id_list_in_this_round,
                    q_u_list=self.coordinator.q_u_list,
                    stepped_q_u_list=self.coordinator.stepped_q_u_list,
                )
                self.aggregator.results["local_loss_diff"].append(loss_diff_dct)
                self.aggregator.consume_dp_for_train_loss_metric(
                    q_u_list=self.coordinator.q_u_list,
                    stepped_q_u_list=self.coordinator.stepped_q_u_list,
                )
                self.coordinator.online_optimize(
                    loss_diff_dct,
                    with_momentum=self.with_momentum,
                    beta=self.momentum_weight,
                )
                logger.info(f"Next HP: (Q_u, C_u) = {self.coordinator.hp_dct_by_eps}")

            logger.info(
                "\n\n========== end {}-th round training ===========\n".format(
                    self.round_idx
                )
            )
            self.round_idx += 1

        if self.agg_strategy in METHOD_GROUP_ULDP_GROUPS:
            self.aggregator.results["privacy_info"] = {
                "group_k": self.group_k,
                "history": [
                    (silo_id, local_trainer.privacy_engine.accountant.history)
                    for silo_id, local_trainer in self.local_trainer_per_silos.items()
                ],
            }

        if self.agg_strategy == METHOD_PULDP_AVG:
            final_loss = []
            final_metric = []
            for silo_id, local_trainer in self.local_trainer_per_silos.items():
                sum_loss, sum_metric = local_trainer.train_loss(
                    round_idx=self.round_idx
                )
                final_loss.append(sum_loss)
                final_metric.append(sum_metric)
            self.aggregator.results["train_loss"] = np.mean(final_loss)
            self.aggregator.results["train_metric"] = np.mean(final_metric)

        if self.agg_strategy == METHOD_PULDP_AVG_ONLINE_TRAIN:
            self.aggregator.results["final_eps"] = {}
            for eps_u, eps_u_list in self.coordinator.epsilon_groups.items():
                final_eps = self.aggregator.accountant_dct[eps_u].get_epsilon(
                    delta=self.aggregator.delta
                )
                logger.info(
                    f"Final epsilon for {eps_u} is {final_eps} (including eps_u)"
                )
                self.aggregator.results["final_eps"][eps_u] = final_eps

        logger.info("Finish federated learning simulation")

    def get_results(self) -> Dict:
        results = dict()
        results["global"] = self.aggregator.get_results()

        if self.agg_strategy in METHOD_GROUP_ULDP_GROUPS:
            results["privacy_info"] = self.aggregator.results["privacy_info"]

        if self.agg_strategy == METHOD_PULDP_AVG:
            results["qC"] = {
                "q_u": self.coordinator.q_u,
                "C_u": self.local_trainer_per_silos[0].C_u,
            }
            results["train"] = {
                "train_loss": self.aggregator.results["train_loss"],
                "train_metric": self.aggregator.results["train_metric"],
            }

        if self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
            results["param_history"] = self.coordinator.param_history
            results["loss_history"] = self.coordinator.loss_history
            results["final_eps"] = self.aggregator.results["final_eps"]

        return results


def build_loss_callback() -> Callable:
    def loss_callback(loss):
        if torch.isnan(loss):
            raise TrainNanError("Stop because Loss is NaN")

    return loss_callback


class TrainNanError(Exception):
    def __init__(self, message):
        super().__init__(message)
