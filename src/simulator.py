from typing import List, Tuple, Dict
import torch
import copy

from aggregator import Aggregator
from local_trainer import ClassificationTrainer
from mylogger import logger


class FLSimulator:
    def __init__(
        self,
        seed: int,
        model: torch.nn.Module,
        train_dataset: List[Tuple[torch.Tensor, int]],
        test_dataset: List[Tuple[torch.Tensor, int]],
        local_dataset_per_silos: Dict[int, List[Tuple[torch.Tensor, int]]],
        n_silos: int,
        device: str,
        n_total_round: int,
        n_silo_per_round: int,
        lr: float,
        local_batch_size: int,
        weight_decay: float,
        client_optimizer: str,
        epochs: int,
        agg_strategy: str,
        clipping_bound: float = None,
        sigma: float = None,
        delta: float = None,
    ):
        self.n_total_round = n_total_round
        self.round_idx = 0
        model.to(device)

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

        self.local_trainer_per_silos: Dict[int, ClassificationTrainer] = {}
        for silo_id, (
            local_train_dataset,
            local_test_dataset,
        ) in local_dataset_per_silos.items():
            local_trainer = ClassificationTrainer(
                base_seed=seed,
                model=copy.deepcopy(model),
                silo_id=silo_id,
                agg_strategy=agg_strategy,
                device=device,
                local_train_dataset=local_train_dataset,
                local_test_dataset=local_test_dataset,
                client_optimizer=client_optimizer,
                learning_rate=lr,
                local_batch_size=local_batch_size,
                weight_decay=weight_decay,
                epochs=epochs,
                local_sigma=sigma,
                local_delta=delta,
                local_clipping_bound=clipping_bound,
            )
            self.local_trainer_per_silos[silo_id] = local_trainer

    def run(self):
        logger.info("Start federated learning simulation")

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
