import copy
from typing import Dict
import unittest
from pathlib import Path
import os
import sys
import time
import numpy as np
from phe import paillier
import logging

logging.disable(logging.CRITICAL)

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)
path_project = str(Path(__file__).resolve().parent.parent.parent)

import models
from dataset import load_dataset
from secure_aggregation import (
    SecureAggregator,
    SecureLocalTrainer,
    PRIMARY_SILO_ID,
    gen_random_int_in_GFp,
    non_recursive_mod_inverse,
    integerize,
    encode,
    decode,
    re_integerize,
    PRECISION,
    DIVISIBLE_NUM,
)


class TestSecureAggregation(unittest.TestCase):
    def test_compute_with_paillier(self):
        pk, sk = paillier.generate_paillier_keypair(n_length=3072)
        modulus = pk.n
        pk.max_int = modulus - 1
        r = gen_random_int_in_GFp(pk.max_int, np.random.RandomState())
        N = 900
        n = 90
        w = 0.01234567890123456789
        w_list = np.array([w, -w]).astype(object)
        int_w_list = integerize(w_list, PRECISION)
        int_w_list = encode(int_w_list, modulus)

        a = non_recursive_mod_inverse(N * r, modulus)
        enc_a = pk.encrypt(a)
        enc_coef = enc_a * DIVISIBLE_NUM * n * r
        enc_out = int_w_list * enc_coef

        out = (
            np.vectorize(lambda x: decode(sk.decrypt(x), modulus))(enc_out)
            / DIVISIBLE_NUM
        )
        self.assertTrue(
            np.abs(
                re_integerize(out, PRECISION)
                - np.array([0.001234567890123456789, -0.001234567890123456789])
            ).max()
            < 1e-7,
        )

    def test_end_to_end_secure_weighted_training(self):
        seed = 0
        dataset_name = "heart_disease"
        model = models.create_model("cnn", dataset_name, seed)
        n_users = 2
        n_silos = 4
        n_silo_per_round = 4
        device = "cpu"
        agg_strategy = "ULDP-AVG-w"
        local_epochs = 1
        data_random_state = np.random.RandomState(seed=seed)
        train_dataset, test_dataset, local_dataset_per_silos = load_dataset(
            data_random_state,
            dataset_name,
            path_project,
            n_users,
            n_silos,
            "uniform",
            "uniform",
            1.0,
            1.0,
            1,
            is_simulation=True,
        )
        secure_aggregator = SecureAggregator(
            model=copy.deepcopy(model),
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_users=n_users,
            n_silos=n_silos,
            n_silo_per_round=n_silo_per_round,
            device=device,
            base_seed=seed,
            strategy=agg_strategy,
            global_learning_rate=1.0,
            clipping_bound=1.0,
            sigma=1.0,
            delta=1e-5,
            dataset_name=dataset_name,
            sampling_rate_q=0.2,
        )

        local_trainer_per_silos: Dict[int, SecureLocalTrainer] = {}
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
                client_optimizer="sgd",
                local_clipping_bound=1.0,
                local_sigma=1.0,
                local_delta=1e-5,
                local_learning_rate=0.001,
                local_epochs=local_epochs,
                n_silo_per_round=n_silo_per_round,
                dataset_name=dataset_name,
            )
            local_trainer_per_silos[silo_id] = local_trainer

        # Protocol Test

        channel_server_to_silos = {silo_id: {} for silo_id in range(n_silos)}
        channel_silo_to_server = {silo_id: {} for silo_id in range(n_silos)}

        start_time = time.time()

        # Step 1: key exchange
        print("\nStep 1: key exchange")

        # silo -> server
        for silo_id in range(n_silos):
            channel_silo_to_server[silo_id] = local_trainer_per_silos[
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
            local_trainer_per_silos[silo_id].receive_dh_pubkeys_and_gen_shared_keys(
                other_silo_pubkeys
            )
            local_trainer_per_silos[silo_id].receive_paillier_public_key(
                paillier_public_key
            )

        # prepare shared random seed
        # silo-0 -> server
        shared_random_seed_per_silo = local_trainer_per_silos[
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
                local_trainer_per_silos[
                    silo_id
                ].receive_across_silos_shared_random_seed(
                    channel_server_to_silos[silo_id]
                )

        print(time.time() - start_time, "sec")

        # Step 2: generate multiplicative blinding masks
        # Step 3: Calculate multiplicative blinded histogram
        # Step 4: Secure aggregation for blinded histogram
        print(
            "Step 2-4: generate multiplicative blinding masks, calculate multiplicative blinded histogram, secure aggregation for blinded histogram"
        )

        # silo -> server
        for silo_id in range(n_silos):
            channel_silo_to_server[silo_id] = local_trainer_per_silos[
                silo_id
            ].make_blinded_user_hist()
            secure_aggregator.receive_blinded_user_histogram(
                silo_id, channel_silo_to_server[silo_id]
            )

        # Step 5: Compute inverse of blinded histogram and encrypt them by Paillier public key
        print(
            "Step 5: Compute inverse of blinded histogram and encrypt them by Paillier public key"
        )
        # server -> silo
        encrypted_inversed_blinded_user_histogram = (
            secure_aggregator.get_encrypt_inversed_blinded_user_histogram()
        )
        for silo_id in range(n_silos):
            channel_server_to_silos[silo_id] = encrypted_inversed_blinded_user_histogram
            local_trainer_per_silos[silo_id].receive_encrypted_weights(
                channel_server_to_silos[silo_id]
            )
        print(time.time() - start_time, "sec")

        # Step 6: Local training with weighting with encrypted weights
        # Step 7: Secure aggregation for encrypted weights
        print(
            "Step 6-7: Local training with weighting with encrypted weights, secure aggregation for encrypted weights"
        )

        # silo
        for silo_id in range(n_silos):
            local_trainer = local_trainer_per_silos[silo_id]

            model_delta_list = []
            for user_id in range(n_users):
                udpated_weight = model.state_dict()  # Mock of local trained model
                encrypted_model_delta = local_trainer.secure_weighting(
                    udpated_weight, user_id
                )
                model_delta_list.append(encrypted_model_delta)

            summed_encrypted_model_delta = model_delta_list[0]
            for i in range(1, len(model_delta_list)):
                for key in encrypted_model_delta.keys():
                    summed_encrypted_model_delta[key] += model_delta_list[i][key]

            encrypted_noised_model_delta = local_trainer._secure_add_noise(
                encrypted_model_delta=summed_encrypted_model_delta,
                sigma=0.0,
                random_state=local_trainer.random_state,
                modulus=local_trainer.modulus,
            )
            masked_model_delta = local_trainer.secagg_mask_params(
                encrypted_noised_model_delta, round_idx=0, modulus=local_trainer.modulus
            )

            channel_silo_to_server[silo_id] = masked_model_delta

        # silo -> server
        for silo_id in range(n_silos):
            secure_aggregator.add_local_trained_result(
                silo_id,
                channel_silo_to_server[silo_id],
                0,
            )

        # # server
        global_weights = secure_aggregator.aggregate(list(range(n_silos)), 0)

        # Correctness Check
        for key in global_weights.keys():
            diff = global_weights[key] - model.state_dict()[key]
            abs_diff = np.abs(diff * n_silos - model.state_dict()[key])
            self.assertTrue(
                (abs_diff.max() < 1e-7).item(),
                "The error is too large and aggregation might not be correct",
            )

        # Correctness Check (noise)
        # if given noise
        # for key in global_weights.keys():
        #     diff = (global_weights[key] - noise[key]/(n_users*n_silos)) - model.state_dict()[key]
        #     abs_diff = np.abs(diff * n_silos  - model.state_dict()[key])
        #     self.assertTrue(
        #         (abs_diff.max() < 1e-7).item(),
        #         "The error is too large and aggregation might not be correct",
        #     )


if __name__ == "__main__":
    unittest.main()
