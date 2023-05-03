from typing import List, Optional, Tuple, Dict
import torch
import copy

from aggregator import Aggregator
from coordinator import Coordinator
from local_trainer import ClassificationTrainer
from mylogger import logger


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
    ):
        self.n_total_round = n_total_round
        self.round_idx = 0
        model.to(device)
        self.agg_strategy = agg_strategy

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
        )

        if self.agg_strategy in ["ULDP-GROUP"]:
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
            )
            self.local_trainer_per_silos[silo_id] = local_trainer
            if self.agg_strategy in ["ULDP-GROUP"]:
                self.coordinator.original_user_hist_dct[silo_id] = user_hist

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
                    self.round_idx
                )
                local_trainer.test_local(self.round_idx)
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
            self.aggregator.test_global(self.round_idx)
            logger.info(
                "\n\n========== end {}-th round training ===========\n".format(
                    self.round_idx
                )
            )
            self.round_idx += 1

        logger.info("Finish federated learning simulation")

    def get_results(self) -> Dict:
        results = dict()
        results["global"] = self.aggregator.get_results()
        results["local"] = dict()
        for silo_id, lt in self.local_trainer_per_silos.items():
            results["local"][silo_id] = lt.get_results()
        return results
