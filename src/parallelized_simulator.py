from multiprocessing.managers import ValueProxy
from typing import Callable, List, Optional, Tuple, Dict
import torch
import copy
import optuna

from aggregator import Aggregator
from coordinator import Coordinator
from local_trainer import ClassificationTrainer
from mylogger import logger

import multiprocessing


# Heuristic early pruning conditions for hyper-parameter tuning
TEST_ACC_THRESHOLDS = {
    "mnist": (3, 0.11),
    "cifar10": (5, 0.11),
    "cifar100": (5, 0.011),
    "heart_disease": (3, 0.1),
    "tcga_brca": (3, 0.1),
    "isic": (3, 0.1),
}


class FLSimulator:
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
        trial: optuna.Trial = None,
        dataset_name: str = None,
        sampling_rate_q: Optional[float] = None,
    ):
        self.n_total_round = n_total_round
        self.round_idx = 0
        model.to(device)
        self.agg_strategy = agg_strategy
        self.dataset_name = dataset_name
        self.sampling_rate_q = sampling_rate_q

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
        )

        if self.agg_strategy in [
            "ULDP-GROUP",
            "ULDP-SGD",
            "ULDP-SGD-w",
            "ULDP-SGD-s",
            "ULDP-SGD-ws",
            "ULDP-AVG",
            "ULDP-AVG-w",
            "ULDP-AVG-s",
            "ULDP-AVG-ws",
        ]:
            self.coordinator = Coordinator(
                base_seed=seed, n_silos=n_silos, n_users=n_users
            )
            self.group_k = group_k
            self.agg_strategy = agg_strategy

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
            )
            self.local_trainer_per_silos[silo_id] = local_trainer
            if self.agg_strategy in [
                "ULDP-GROUP",
                "ULDP-SGD",
                "ULDP-SGD-w",
                "ULDP-SGD-s",
                "ULDP-AVG",
                "ULDP-AVG-w",
                "ULDP-AVG-s",
            ]:
                self.coordinator.set_user_hist_by_silo_id(silo_id, user_hist)

            self.trial = trial

    def run(self):
        logger.info("Start federated learning simulation")

        if self.agg_strategy == "ULDP-GROUP":
            if self.dataset_name == "tcga_brca":
                min_count = 2
            else:
                min_count = 1
            bounded_user_hist_per_silo = self.coordinator.build_user_bound_histograms(
                self.group_k, self.coordinator.original_user_hist_dct, min_count
            )

            for silo_id, bounded_user_hist in bounded_user_hist_per_silo.items():
                self.local_trainer_per_silos[silo_id].bound_user_contributions(
                    bounded_user_hist
                )
        elif self.agg_strategy in [
            "ULDP-SGD",
            "ULDP-AVG",
            "ULDP-SGD-w",
            "ULDP-AVG-w",
        ]:
            if self.agg_strategy in ["ULDP-SGD-w", "ULDP-AVG-w"]:
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

            if self.agg_strategy in [
                "ULDP-SGD-s",
                "ULDP-AVG-s",
                "ULDP-AVG-ws",
                "ULDP-SGD-ws",
            ]:
                user_weights_per_silo = self.coordinator.build_user_weights(
                    weighted=False, sampling_rate_q=self.sampling_rate_q
                )
                for silo_id, user_weights in user_weights_per_silo.items():
                    self.local_trainer_per_silos[silo_id].set_user_weights(user_weights)

            with multiprocessing.Manager() as manager:
                shared_aggregator = manager.Value(Aggregator, self.aggregator)
                shared_trial = manager.Value(optuna.Trial, self.trial)

                with multiprocessing.get_context("spawn").Pool() as pool:
                    results = [
                        pool.apply_async(
                            train_silo,
                            args=(
                                silo_id,
                                self.round_idx,
                                self.local_trainer_per_silos[silo_id],
                                shared_aggregator,
                                shared_trial,
                            ),
                        )
                        for silo_id in silo_id_list_in_this_round
                    ]

                    # 全てのプロセスの終了を待機
                    for result in results:
                        (
                            silo_id,
                            local_updated_weights,
                            n_local_sample,
                            latest_eps,
                        ) = result.get()
                        self.aggregator.add_local_trained_result(
                            silo_id, local_updated_weights, n_local_sample, latest_eps
                        )

            logger.debug(
                "============ AGGREGATION: ROUND %d ============" % (self.round_idx)
            )
            self.aggregator.aggregate(silo_id_list_in_this_round, self.round_idx)
            test_acc, _ = self.aggregator.test_global(self.round_idx)
            logger.debug(
                "\n\n========== end {}-th round training ===========\n".format(
                    self.round_idx
                )
            )
            self.round_idx += 1

            if self.trial is not None:
                self.trial.report(1.0 - test_acc, self.round_idx)
                if self.trial.should_prune():
                    logger.warning(
                        "PRUNED BECAUSE OF TOO LOW ACCURACY COMPARED TO MEDIAN"
                    )
                    raise optuna.exceptions.TrialPruned()

                threshould = TEST_ACC_THRESHOLDS[self.dataset_name]
                if self.round_idx + 1 >= threshould[0] and test_acc <= threshould[1]:
                    logger.warning("PRUNED BECAUSE OF TOO LOW ACCURACY")
                    raise optuna.exceptions.TrialPruned()

        logger.info("Finish federated learning simulation")

    def get_results(self) -> Dict:
        results = dict()
        results["global"] = self.aggregator.get_results()
        # if self.agg_strategy in ["ULDP-SGD", "ULDP-AVG", "ULDP-SGD-w", "ULDP-AVG-w"]:
        #     pass
        # else:
        #     results["local"] = dict()
        #     for silo_id, lt in self.local_trainer_per_silos.items():
        #         results["local"][silo_id] = lt.get_results()
        return results


def build_loss_callback(trial) -> Callable:
    if trial is None:

        def loss_callback(loss):
            if torch.isnan(loss):
                raise OverflowError("Stop because Loss is NaN")

    else:

        def loss_callback(loss):
            # check if loss is nan
            if torch.isnan(loss):
                logger.warning("PRUNED LOSS IS NAN")
                raise optuna.exceptions.TrialPruned()

    return loss_callback


def train_silo(
    silo_id,
    round_idx,
    local_trainer: ClassificationTrainer,
    shared_aggregator: ValueProxy[Aggregator],
    shared_trial: ValueProxy[optuna.Trial],
):
    logger.debug(
        "============ TRAINING: SILO_ID = %d (ROUND %d) ============"
        % (silo_id, round_idx)
    )
    local_trainer.set_model_params(shared_aggregator.value.get_global_model_params())

    local_updated_weights, n_local_sample = local_trainer.train(
        round_idx, loss_callback=build_loss_callback(shared_trial.value)
    )

    return (
        silo_id,
        local_updated_weights,
        n_local_sample,
        local_trainer.get_latest_epsilon(),
    )
