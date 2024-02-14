from typing import Callable, List, Optional, Tuple, Dict
import torch
import numpy as np
import copy

from aggregator import Aggregator
from dataset import TCGA_BRCA
from method_group import (
    METHOD_GROUP_ENHANCED_WEIGHTED_WITHOUT_SAMPLING,
    METHOD_GROUP_NEED_USER_RECORD,
    METHOD_GROUP_SAMPLING_WITH_ENHANCED_WEIGHTING,
    METHOD_GROUP_SAMPLING_WITHOUT_ENHANCED_WEIGHTING,
    METHOD_GROUP_ULDP_WITHOUT_SAMPLING,
    METHOD_GROUP_ONLINE_OPTIMIZATION,
    METHOD_GROUP_ULDP_GROUPS,
    METHOD_PULDP_AVG,
    METHOD_PULDP_QC_TEST,
    METHOD_PULDP_QC_TRAIN,
)
from coordinator import Coordinator
from local_trainer import ClassificationTrainer
import parallelized_local_trainer
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
        off_train_loss_noise: Optional[bool] = False,
        momentum_weight: Optional[float] = None,
        hp_baseline: Optional[bool] = False,
        step_decay: Optional[bool] = False,
        initial_q_u: Optional[float] = None,
        parallelized: Optional[bool] = False,
        gpu_id: Optional[int] = None,
        dynamic_global_learning_rate: Optional[bool] = False,
    ):
        self.n_total_round = n_total_round
        self.round_idx = 0
        model.to(device)
        self.agg_strategy = agg_strategy
        self.dataset_name = dataset_name
        self.sampling_rate_q = sampling_rate_q
        self.validation_ratio = validation_ratio
        self.hp_baseline = hp_baseline
        self.parallelized = parallelized
        self.device = device
        self.gpu_id = gpu_id
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
            step_decay=step_decay,
            initial_q_u=initial_q_u,
            hp_baseline=hp_baseline,
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
            n_total_round=n_total_round,
            dynamic_global_learning_rate=dynamic_global_learning_rate,
        )

        if self.agg_strategy == METHOD_PULDP_AVG:
            assert (
                q_u is not None and C_u is not None
            ), "q_u and C_u must be provided in PULDP-AVG"
            self.aggregator.sampling_rate_q = np.mean(list(q_u.values()))
            self.aggregator.set_average_qC(
                np.mean(np.array(list(C_u.values())) * np.array(list(q_u.values())))
            )

        if self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
            self.with_momentum = with_momentum
            self.off_train_loss_noise = off_train_loss_noise
            self.momentum_weight = momentum_weight
            self.aggregator.set_epsilon_groups(self.coordinator.epsilon_groups)
        if self.agg_strategy == METHOD_PULDP_QC_TEST:
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
                n_total_round=n_total_round,
                hp_baseline=hp_baseline,
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

        if self.parallelized:
            from multiprocessing import Process, cpu_count, Queue, set_start_method
            import os

            if os.name != "posix" or "linux" in os.uname().sysname.lower():
                # Linux
                # num_workers = 8 # you may need to adjust GPU memory size when getting memory allocation error
                set_start_method("forkserver", force=True)

            input_queue = Queue()
            output_queue = Queue()
            num_workers = min(cpu_count() - 1, self.aggregator.n_silos)

            # Wakeup worker processes pool
            processes = [
                Process(
                    target=parallelized_local_trainer.parallelized_train_worker,
                    args=(input_queue, output_queue),
                )
                for _ in range(num_workers)
            ]
            for p in processes:
                p.start()

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
        elif self.agg_strategy in METHOD_GROUP_ULDP_WITHOUT_SAMPLING:
            if self.agg_strategy in METHOD_GROUP_ENHANCED_WEIGHTED_WITHOUT_SAMPLING:
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

            if self.agg_strategy in METHOD_GROUP_SAMPLING_WITHOUT_ENHANCED_WEIGHTING:
                user_weights_per_silo = self.coordinator.build_user_weights(
                    weighted=False, is_sample=True
                )
                for silo_id, user_weights in user_weights_per_silo.items():
                    self.local_trainer_per_silos[silo_id].set_user_weights(user_weights)

            elif self.agg_strategy in METHOD_GROUP_SAMPLING_WITH_ENHANCED_WEIGHTING:
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
                    round_idx=self.round_idx,
                    accountant_dct=self.aggregator.accountant_dct,
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
                self.aggregator.set_average_qC(
                    np.mean(self.coordinator.C_u_list * self.coordinator.q_u_list)
                )

            for silo_id in silo_id_list_in_this_round:
                logger.debug(
                    "============ TRAINING: SILO_ID = %d (ROUND %d) ============"
                    % (silo_id, self.round_idx)
                )

                local_trainer = self.local_trainer_per_silos[silo_id]
                local_trainer.set_model_params(
                    self.aggregator.get_global_model_params()
                )

                if self.parallelized:  # Parallelized training
                    if self.parallelized == "strong":
                        num_gpu = 4
                        gpu_id = silo_id % num_gpu
                    else:
                        gpu_id = self.gpu_id
                    input_queue.put(
                        (
                            silo_id,
                            gpu_id,
                            local_trainer.random_state,
                            local_trainer.model,
                            local_trainer.device,
                            local_trainer.agg_strategy,
                            local_trainer.train_loader,
                            local_trainer.criterion,
                            self.round_idx,
                            local_trainer.n_silo_per_round,
                            getattr(local_trainer, "privacy_engine", None),
                            getattr(local_trainer, "clipping_bound", None),
                            getattr(local_trainer, "local_sigma", None),
                            getattr(local_trainer, "local_delta", None),
                            getattr(local_trainer, "user_level_data_loader", None),
                            getattr(local_trainer, "user_weights", None),
                            local_trainer.dataset_name,
                            local_trainer.client_optimizer,
                            local_trainer.local_learning_rate,
                            local_trainer.weight_decay,
                            local_trainer.local_epochs,
                            getattr(local_trainer, "C_u", None),
                            getattr(local_trainer, "epsilon_groups", None),
                            getattr(local_trainer, "C_u_list", None),
                            getattr(
                                local_trainer, "stepped_C_u_list_for_optimization", None
                            ),
                            getattr(
                                local_trainer, "user_weights_for_optimization", None
                            ),
                            getattr(
                                local_trainer,
                                "stepped_user_weights_for_optimization",
                                None,
                            ),
                            getattr(local_trainer, "C_u_list_for_optimization", None),
                            getattr(self.coordinator, "q_u_list", None),
                            getattr(self.coordinator, "stepped_q_u_list", None),
                            getattr(self, "off_train_loss_noise", None),
                            getattr(local_trainer, "accountant_dct", None),
                            self.n_total_round,
                            self.hp_baseline,
                        )
                    )
                else:
                    if self.agg_strategy == METHOD_PULDP_QC_TEST:
                        local_updated_weights_dct = local_trainer.train(
                            self.round_idx,
                            loss_callback=build_loss_callback(),
                        )
                        local_trainer.consume_dp_for_model_optimization(
                            self.coordinator.q_u_list
                        )
                        local_trainer.consume_dp_for_model_optimization(
                            self.coordinator.q_u_list
                        )
                        local_trainer.consume_dp_for_stepped_model_optimization(
                            self.coordinator.stepped_q_u_list
                        )
                        self.aggregator.add_local_trained_result_of_QCTest(
                            silo_id,
                            local_updated_weights_dct,
                            local_trainer.get_latest_epsilon(),
                        )
                    elif self.agg_strategy == METHOD_PULDP_QC_TRAIN:
                        local_updated_weights_dct = local_trainer.train(
                            self.round_idx,
                            loss_callback=build_loss_callback(),
                        )
                        local_trainer.consume_dp_for_model_optimization(
                            q_u_list=self.coordinator.q_u_list
                        )
                        if local_trainer.hp_baseline:
                            local_loss_diff_dct = {}
                        else:
                            local_loss_diff_dct = local_trainer.compute_train_loss_for_online_optimization_and_consume_dp_for_train_loss_metric(
                                self.round_idx,
                                self.coordinator.q_u_list,
                                self.coordinator.stepped_q_u_list,
                                local_updated_weights_dct,
                                off_train_loss_noise=self.off_train_loss_noise,
                            )
                        self.aggregator.add_local_trained_result_of_QCTrain(
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

            if self.parallelized:
                expected_results = len(silo_id_list_in_this_round)
                received_results = 0

                while received_results < expected_results:
                    silo_id, random_state, accountant_dct, result = output_queue.get()
                    if self.agg_strategy == METHOD_PULDP_QC_TEST:
                        local_updated_weights_dct = result
                        self.aggregator.add_local_trained_result_of_QCTest(
                            silo_id,
                            local_updated_weights_dct,
                            0.0,
                        )
                    elif self.agg_strategy == METHOD_PULDP_QC_TRAIN:
                        local_updated_weights_dct, local_loss_diff_dct = result
                        self.aggregator.add_local_trained_result_of_QCTrain(
                            silo_id,
                            local_updated_weights_dct["default"],
                            0.0,
                            local_loss_diff_dct,
                        )
                    else:
                        local_updated_weights, _ = result
                        self.aggregator.add_local_trained_result(
                            silo_id,
                            local_updated_weights,
                            0.0,
                        )
                    local_trainer = self.local_trainer_per_silos[silo_id]
                    local_trainer.random_state = random_state
                    local_trainer.accountant_dct = accountant_dct

                    received_results += 1

            logger.debug(
                "============ AGGREGATION: ROUND %d ============" % (self.round_idx)
            )

            if self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
                self.aggregator.consume_dp_for_model_optimization(
                    q_u_list=self.coordinator.q_u_list,
                    C_u_list=self.coordinator.C_u_list,
                )
                if self.agg_strategy == METHOD_PULDP_QC_TEST:
                    self.aggregator.consume_dp_for_model_optimization(
                        self.coordinator.q_u_list, self.coordinator.C_u_list
                    )
                    self.aggregator.consume_dp_for_stepped_model_optimization(
                        self.coordinator.stepped_q_u_list,
                        self.coordinator.stepped_C_u_list,
                    )
                    loss_diff_dct = self.aggregator.compute_loss_diff(
                        silo_id_list_in_this_round,
                        q_u_list=self.coordinator.q_u_list,
                        stepped_q_u_list=self.coordinator.stepped_q_u_list,
                        C_u_list=self.coordinator.C_u_list,
                        stepped_C_u_list=self.coordinator.stepped_C_u_list,
                    )
                elif self.agg_strategy == METHOD_PULDP_QC_TRAIN:
                    if self.hp_baseline:
                        loss_diff_dct = {}
                    else:
                        self.aggregator.consume_dp_for_train_loss_metric(
                            q_u_list=self.coordinator.q_u_list,
                            stepped_q_u_list=self.coordinator.stepped_q_u_list,
                            round_idx=self.round_idx,
                        )
                        loss_diff_dct = self.aggregator.compute_loss_diff(
                            silo_id_list_in_this_round,
                            q_u_list=self.coordinator.q_u_list,
                            stepped_q_u_list=self.coordinator.stepped_q_u_list,
                        )

                self.aggregator.results["local_loss_diff"].append(loss_diff_dct)
                # until last round
                if self.round_idx + 1 < self.n_total_round:
                    self.coordinator.online_optimize(
                        loss_diff_dct,
                        with_momentum=self.with_momentum,
                        beta=self.momentum_weight,
                        round_idx=self.round_idx,
                        accountant_dct=self.aggregator.accountant_dct,
                    )
                    logger.info(
                        f"Next HP: (Q_u, C_u) = {self.coordinator.hp_dct_by_eps}"
                    )

            self.aggregator.aggregate(silo_id_list_in_this_round, self.round_idx)
            self.aggregator.test_global(self.round_idx)

            if self.validation_ratio > 0.0:
                self.aggregator.test_global(self.round_idx, is_validation=True)

            logger.info(
                "\n\n========== end {}-th round training ===========\n".format(
                    self.round_idx
                )
            )
            self.round_idx += 1

            if self.parallelized and self.gpu_id is not None:
                # when using GPU, we need to restart the process pool to avoid memory leak
                for _ in range(num_workers):
                    input_queue.put(None)
                for p in processes:
                    p.join()

                del processes

                processes = [
                    Process(
                        target=parallelized_local_trainer.parallelized_train_worker,
                        args=(input_queue, output_queue),
                    )
                    for _ in range(num_workers)
                ]
                for p in processes:
                    p.start()

        if self.parallelized:
            # messaging finish to worker processes
            for _ in range(num_workers):
                input_queue.put(None)

            # wait for all worker processes to finish
            for p in processes:
                p.join()

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

        if self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
            self.aggregator.results["final_eps"] = {}
            for eps_u, eps_u_list in self.coordinator.epsilon_groups.items():
                final_eps = self.aggregator.accountant_dct[eps_u].get_epsilon(
                    delta=self.aggregator.delta
                )
                assert (
                    final_eps <= eps_u
                ), f"Final epsilon {final_eps} must be less than eps_u"
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
            results["q_u"] = self.coordinator.q_u
            results["C_u"] = self.local_trainer_per_silos[0].C_u
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
