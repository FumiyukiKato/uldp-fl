from typing import Callable, List, Optional, Tuple, Dict
import torch
import copy
import optuna

from aggregator import Aggregator
from coordinator import Coordinator
from local_trainer import ClassificationTrainer
from mylogger import logger


# Heuristic early pruning conditions for hyper-parameter tuning
TEST_ACC_THRESHOLDS = {
    "mnist": (3, 0.11),
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
        learning_rate: float,
        local_batch_size: int,
        weight_decay: float,
        client_optimizer: str,
        epochs: int,
        agg_strategy: str,
        clipping_bound: Optional[float] = None,
        sigma: Optional[float] = None,
        delta: Optional[float] = None,
        group_k: Optional[int] = None,
        trial: optuna.Trial = None,
        dataset_name: str = None,
    ):
        self.n_total_round = n_total_round
        self.round_idx = 0
        model.to(device)
        self.agg_strategy = agg_strategy
        self.dataset_name = dataset_name

        self.aggregator = Aggregator(
            model=copy.deepcopy(model),
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_silos=n_silos,
            n_silo_per_round=n_silo_per_round,
            device=device,
            base_seed=seed,
            strategy=agg_strategy,
            clipping_bound=clipping_bound,
            sigma=sigma,
            delta=delta,
            central_learning_rate=learning_rate,
        )

        if self.agg_strategy in ["ULDP-GROUP", "ULDP-SGD", "ULDP-AVG"]:
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
                learning_rate=learning_rate,
                local_batch_size=local_batch_size,
                weight_decay=weight_decay,
                epochs=epochs,
                local_sigma=sigma,
                local_delta=delta,
                local_clipping_bound=clipping_bound,
                group_k=group_k,
                n_silo_per_round=n_silo_per_round,
            )
            self.local_trainer_per_silos[silo_id] = local_trainer
            if self.agg_strategy in ["ULDP-GROUP", "ULDP-SGD", "ULDP-AVG"]:
                self.coordinator.original_user_hist_dct[silo_id] = user_hist

            self.trial = trial

    def run(self):
        logger.info("Start federated learning simulation")

        if self.agg_strategy == "ULDP-GROUP":
            bounded_user_hist_per_silo = self.coordinator.build_user_bound_histograms(
                self.group_k, self.coordinator.original_user_hist_dct
            )
            for silo_id, bounded_user_hist in bounded_user_hist_per_silo.items():
                self.local_trainer_per_silos[silo_id].bound_user_contributions(
                    bounded_user_hist
                )
        elif self.agg_strategy in ["ULDP-SGD", "ULDP-AVG"]:
            user_weights_per_silo = self.coordinator.build_user_weights(uniform=True)
            for silo_id, user_weights in user_weights_per_silo.items():
                self.local_trainer_per_silos[silo_id].set_user_weights(user_weights)

        while self.round_idx < self.n_total_round:
            silo_id_list_in_this_round = self.aggregator.silo_selection()
            for silo_id in silo_id_list_in_this_round:
                logger.info(
                    "============ TRAINING: SILO_ID = %d (ROUND %d) ============"
                    % (silo_id, self.round_idx)
                )
                local_trainer = self.local_trainer_per_silos[silo_id]
                local_trainer.set_model_params(
                    self.aggregator.get_global_model_params()
                )

                local_updated_weights, n_local_sample = local_trainer.train(
                    self.round_idx,
                    loss_callback=build_loss_callback(self.trial),
                )
                local_trainer.test_local(self.round_idx)
                if self.agg_strategy in [
                    "DEFAULT",
                    "ULDP-AVG",
                    "ULDP-GROUP",
                    "ULDP-NAIVE",
                ]:
                    # test local model with global test dataset
                    self.aggregator.test_global(
                        self.round_idx, model=local_trainer.model, silo_id=silo_id
                    )
                self.aggregator.add_local_trained_result(
                    silo_id,
                    local_updated_weights,
                    n_local_sample,
                    local_trainer.get_latest_epsilon(),
                )

            logger.info(
                "============  AGGREGATION: ROUND %d ============" % (self.round_idx)
            )
            self.aggregator.aggregate(silo_id_list_in_this_round, self.round_idx)
            test_acc, _ = self.aggregator.test_global(self.round_idx)
            logger.info(
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

    def set_learning_rate(self, learning_rate):
        self.aggregator.central_learning_rate = learning_rate
        for local_trainer in self.local_trainer_per_silos.values():
            local_trainer.learning_rate = learning_rate

    def set_sigma(self, sigma):
        self.aggregator.sigma = sigma
        for local_trainer in self.local_trainer_per_silos.values():
            local_trainer.local_sigma = sigma

    def set_clipping_bound(self, clipping_bound: float):
        self.aggregator.clipping_bound = clipping_bound
        for local_trainer in self.local_trainer_per_silos.values():
            local_trainer.local_clipping_bound = clipping_bound

    def get_results(self) -> Dict:
        results = dict()
        results["global"] = self.aggregator.get_results()
        # if self.agg_strategy in ["ULDP-SGD", "ULDP-AVG"]:
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
                raise ValueError("Stop because Loss is NaN")

    else:

        def loss_callback(loss):
            # check if loss is nan
            if torch.isnan(loss):
                logger.warning("PRUNED LOSS IS NAN")
                raise optuna.exceptions.TrialPruned()

    return loss_callback
