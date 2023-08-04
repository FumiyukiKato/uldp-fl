import time
from typing import List, Optional, Tuple, Dict
import torch
import copy

from secure_aggregation import SecureAggregator, SecureLocalTrainer, PRIMARY_SILO_ID
from mylogger import logger


class SecureWeightingFLSimulator:
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
    ):
        self.n_total_round = n_total_round
        self.round_idx = 0
        model.to(device)
        self.agg_strategy = agg_strategy
        if agg_strategy not in [
            "ULDP-AVG-w",
            "ULDP-AVG-ws",
            "ULDP-SGD-w",
            "ULDP-SGD-ws",
        ]:
            raise ValueError(f"agg_strategy {agg_strategy} is not supported.")

        self.dataset_name = dataset_name
        self.sampling_rate_q = sampling_rate_q
        self.n_silos = n_silos
        self.n_users = n_users

        self.time_results = {"round_idx": [], "time": [], "kind": [], "counter": []}
        self.time_counter = 0

        self.secure_aggregator = SecureAggregator(
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

        self.local_trainer_per_silos: Dict[int, SecureLocalTrainer] = {}
        for silo_id, (
            local_train_dataset,
            local_test_dataset,
            user_hist,
            user_ids,
        ) in local_dataset_per_silos.items():
            local_trainer = SecureLocalTrainer(
                base_seed=seed,
                model=copy.deepcopy(model),
                silo_id=silo_id,
                agg_strategy=agg_strategy,
                device=device,
                local_train_dataset=local_train_dataset,
                local_test_dataset=local_test_dataset,
                user_histogram=user_hist,
                user_ids_of_local_train_dataset=user_ids,
                n_users=n_users,
                n_silos=n_silos,
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

    def record_time(self, start_time, kind: str):
        this_time = time.time()
        print(kind, this_time - self.previous_time)
        self.time_results["round_idx"].append(self.round_idx)
        self.time_results["time"].append(this_time - start_time)
        self.time_results["kind"].append(kind)
        self.time_results["counter"].append(self.time_counter)
        self.time_counter += 1
        self.previous_time = this_time

    def run(self):
        logger.info("Start federated learning simulation with secure weighting.")
        start_time = time.time()
        self.previous_time = start_time

        n_silos = self.n_silos
        secure_aggregator = self.secure_aggregator

        # mock up the communicatoin channel
        channel_server_to_silos = {silo_id: {} for silo_id in range(n_silos)}
        channel_silo_to_server = {silo_id: {} for silo_id in range(n_silos)}

        # -- Step 1: key exchange
        logger.debug("key exchange")

        # silo -> server
        for silo_id in range(n_silos):
            channel_silo_to_server[silo_id] = self.local_trainer_per_silos[
                silo_id
            ].get_dh_pubkey()
            secure_aggregator.receive_dh_pubkey(
                silo_id, channel_silo_to_server[silo_id]
            )

        # server -> silo
        for silo_id in range(n_silos):
            channel_server_to_silos[silo_id] = (
                secure_aggregator.client_keys,
                secure_aggregator.paillier_public_key,
            )
        # silo
        for silo_id in range(n_silos):
            other_silo_pubkeys, paillier_public_key = channel_server_to_silos[silo_id]
            self.local_trainer_per_silos[
                silo_id
            ].receive_dh_pubkeys_and_gen_shared_keys(other_silo_pubkeys)
            self.local_trainer_per_silos[silo_id].receive_paillier_public_key(
                paillier_public_key
            )

        # prepare shared random seed
        # silo-0 -> server
        shared_random_seed_per_silo = self.local_trainer_per_silos[
            PRIMARY_SILO_ID
        ].gen_across_silos_shared_random_seed()
        channel_silo_to_server[PRIMARY_SILO_ID] = shared_random_seed_per_silo
        secure_aggregator.receive_shared_random_seed(
            channel_silo_to_server[PRIMARY_SILO_ID]
        )

        # server -> silo
        for silo_id in range(n_silos):
            if silo_id != PRIMARY_SILO_ID:
                channel_server_to_silos[
                    silo_id
                ] = secure_aggregator.get_shared_random_seed()[silo_id]
                self.local_trainer_per_silos[
                    silo_id
                ].receive_across_silos_shared_random_seed(
                    channel_server_to_silos[silo_id]
                )

        self.record_time(start_time, "key_exchange")

        # -- Step 2: generate multiplicative blinding masks
        # -- Step 3: Calculate multiplicative blinded histogram
        # -- Step 4: Secure aggregation for blinded histogram and Compute inverse of blinded histogram
        logger.debug(
            "generate multiplicative blinding masks, calculate multiplicative blinded histogram, secure aggregation for blinded histogram"
        )

        # silo -> server
        for silo_id in range(n_silos):
            channel_silo_to_server[silo_id] = self.local_trainer_per_silos[
                silo_id
            ].make_blinded_user_hist()
            secure_aggregator.receive_blinded_user_histogram(
                silo_id, channel_silo_to_server[silo_id]
            )

        if self.agg_strategy in ["ULDP-SGD-w", "ULDP-AVG-w"]:
            # server -> silo
            encrypted_inversed_blinded_user_histogram = (
                secure_aggregator.get_encrypt_inversed_blinded_user_histogram()
            )
            for silo_id in range(n_silos):
                channel_server_to_silos[
                    silo_id
                ] = encrypted_inversed_blinded_user_histogram
                self.local_trainer_per_silos[silo_id].receive_encrypted_weights(
                    channel_server_to_silos[silo_id]
                )

        self.record_time(start_time, "multiplicative_blind_user_hist")

        # start round
        while self.round_idx < self.n_total_round:
            silo_id_list_in_this_round = secure_aggregator.silo_selection()

            if self.agg_strategy in ["ULDP-SGD-ws", "ULDP-AVG-ws"]:
                # server -> silo
                encrypted_inversed_blinded_user_histogram = (
                    secure_aggregator.get_encrypt_inversed_blinded_user_histogram_with_userlevel_subsampling()
                )
                for silo_id in range(n_silos):
                    channel_server_to_silos[
                        silo_id
                    ] = encrypted_inversed_blinded_user_histogram
                    self.local_trainer_per_silos[silo_id].receive_encrypted_weights(
                        channel_server_to_silos[silo_id]
                    )
                self.record_time(
                    start_time, "multiplicative_blind_user_hist_with_subsampling"
                )

            for silo_id in silo_id_list_in_this_round:
                logger.debug(
                    "============ TRAINING: SILO_ID = %d (ROUND %d) ============"
                    % (silo_id, self.round_idx)
                )
                local_trainer = self.local_trainer_per_silos[silo_id]
                local_trainer.set_model_params(
                    secure_aggregator.get_global_model_params()
                )

                local_updated_weights = local_trainer.train(self.round_idx)

                secure_aggregator.add_local_trained_result(
                    silo_id, local_updated_weights, 0, 0
                )
                self.record_time(start_time, f"training_silo_{silo_id}")

            logger.debug(
                "============ AGGREGATION: ROUND %d ============" % (self.round_idx)
            )
            secure_aggregator.aggregate(silo_id_list_in_this_round, self.round_idx)
            self.record_time(start_time, "aggregation")

            test_acc, _ = secure_aggregator.test_global(self.round_idx)
            self.record_time(start_time, "global_test")
            logger.info(
                "\n\n========== end {}-th round training ===========\n".format(
                    self.round_idx
                )
            )
            self.round_idx += 1

        self.record_time(start_time, "total")
        logger.info("Finish federated learning simulation")

    def get_results(self) -> Dict:
        results = dict()
        results["global"] = self.secure_aggregator.get_results()
        results["time"] = self.time_results
        return results
