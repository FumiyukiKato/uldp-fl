import copy
import math
from typing import Dict, List, Optional, OrderedDict, Tuple
import numpy as np
from phe import paillier
import phe
import torch
import pyDH
import hashlib

from aggregator import Aggregator
from local_trainer import ClassificationTrainer


N_LENGTH = 3072
PRECISION = 1e-10  # considering python's float64 precision, ~ 1e-15 is the appropriate precision
MAX_N_USER = 1000
PRIMARY_SILO_ID = 0


# Extended Euclidean Algorithm
def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return gcd, y - (b // a) * x, x


def mod_inverse(a, m):
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise Exception("Modular inverse does not exist")
    else:
        return x % m


# Iterative Extended Euclidean Algorithm
def non_recursive_extended_gcd(a, b):
    x, y, u, v = 0, 1, 1, 0
    while a != 0:
        q, r = b // a, b % a
        m, n = x - u * q, y - v * q
        b, a, x, y, u, v = a, r, u, v, m, n
    gcd = b
    return gcd, x, y


def non_recursive_mod_inverse(a, m):
    gcd, x, _ = non_recursive_extended_gcd(a, m)
    if gcd != 1:
        raise Exception("Modular inverse does not exist")
    else:
        return x % m


def random_in_GFp(p: int, random_state: np.random.RandomState):
    n_length = p.bit_length()
    while True:
        bits = random_state.randint(0, 2, n_length)
        random_number = int("".join(map(str, bits)), 2)
        if random_number < p:
            return random_number


def integerize(val: np.ndarray, precision: int = PRECISION) -> np.ndarray:
    int_val = (val / precision).astype(np.int64)
    bottomupped_int_val = int_val.astype(object)
    return bottomupped_int_val


def re_integerize(int_val: np.ndarray, precision: int = PRECISION) -> np.ndarray:
    bottomupped_float_val = int_val * precision
    return bottomupped_float_val.astype(np.float64)


def encode(value, modulus):
    return value % modulus


def decode(value, modulus):
    if value > modulus // 2:
        return value - modulus
    else:
        return value


def get_perfect_divisible_number(max: int) -> int:
    numbers = range(1, max + 1)
    lcm = math.lcm(*numbers)
    return lcm


DIVISIBLE_NUM = get_perfect_divisible_number(MAX_N_USER)


class SecureAggregator(Aggregator):
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        n_users,
        n_silos,
        n_silo_per_round,
        device,
        base_seed: int,
        strategy: str,
        clipping_bound: Optional[float] = None,
        sigma: Optional[float] = None,
        delta: Optional[float] = None,
        global_learning_rate: Optional[float] = None,
        dataset_name: str = None,
        sampling_rate_q: Optional[float] = None,
    ):
        super().__init__(
            model,
            train_dataset,
            test_dataset,
            n_users,
            n_silos,
            n_silo_per_round,
            device,
            base_seed,
            strategy,
            clipping_bound,
            sigma,
            delta,
            global_learning_rate,
            dataset_name,
            sampling_rate_q,
        )
        self.dh_exchange_ready_client_ids = set()
        self.client_keys = dict()
        self.histogram_ready_client_ids = set()
        self.blinded_user_histogram = {user_id: 0 for user_id in range(self.n_users)}

        (
            self.paillier_public_key,
            self.paillier_private_key,
        ) = paillier.generate_paillier_keypair(n_length=N_LENGTH)
        self.paillier_public_key.max_int = self.paillier_public_key.n - 1
        self.modulus = self.paillier_public_key.n

    def receive_dh_pubkey(self, silo_id: int, sent_dh_pubkey: int) -> Dict[int, int]:
        self.client_keys[silo_id] = sent_dh_pubkey
        self.dh_exchange_ready_client_ids.add(silo_id)
        if len(self.dh_exchange_ready_client_ids) == self.n_silo_per_round:
            return self.client_keys

    def receive_user_records_histogram(
        self, silo_id: int, sent_blinded_user_records_histogram: Dict[int, int]
    ) -> Dict[int, int]:
        for user_id, blinded_count in sent_blinded_user_records_histogram.items():
            self.blinded_user_histogram[user_id] += blinded_count
            self.blinded_user_histogram[user_id] %= self.modulus
        self.histogram_ready_client_ids.add(silo_id)
        if len(self.histogram_ready_client_ids) == self.n_silo_per_round:
            return self.blinded_user_histogram

    def receive_shared_random_seed(
        self, sent_shared_random_seed_per_silo: Dict[int, int]
    ):
        self.shared_random_seed_per_silo = sent_shared_random_seed_per_silo

    def get_shared_random_seed(self) -> Dict[int, int]:
        return self.shared_random_seed_per_silo

    def compute_inverse(self) -> Dict[int, int]:
        inversed_blinded_user_histogram = dict()
        for user_id, blinded_count in self.blinded_user_histogram.items():
            inversed_blinded_user_histogram[user_id] = non_recursive_mod_inverse(
                blinded_count, self.modulus
            )
        return inversed_blinded_user_histogram

    def user_level_subsampling(
        self, inversed_blinded_user_histogram: Dict[int, int], sampling_rate_q: float
    ):
        inversed_blinded_user_histogram = dict()
        user_ids = np.array(range(self.n_users))
        sampled_user_ids = user_ids[
            self.random_state.rand(len(user_ids)) < sampling_rate_q
        ]
        sampled_user_ids_set = set(sampled_user_ids)
        for silo_id in range(self.n_silos):
            for user_id in range(self.n_users):
                if user_id not in sampled_user_ids_set:
                    inversed_blinded_user_histogram[silo_id][user_id] = 0

    def encrypt_inversed_blinded_user_histogram(
        self, inversed_blinded_user_histogram: Dict[int, int]
    ) -> Dict[int, phe.EncryptedNumber]:
        encrypted_weights = dict()
        for user_id, value in inversed_blinded_user_histogram.items():
            encrypted_weights[user_id] = self.paillier_public_key.encrypt(value)
        return encrypted_weights

    def add_local_trained_result(self, silo_id, model_params, n_sample, eps):
        self.model_dict[silo_id] = model_params
        self.n_sample_dict[silo_id] = n_sample
        self.flag_client_model_uploaded_dict[silo_id] = True
        self.latest_eps = max(self.latest_eps, eps)

    def _aggregate_with_decrypt(
        self, param_list: List[OrderedDict], n_avg: float
    ) -> OrderedDict:
        aggregated_params = copy.deepcopy(param_list[0])
        for k in aggregated_params.keys():
            for i in range(1, len(param_list)):
                aggregated_params[k] += param_list[i][k]

        for key in aggregated_params.keys():
            aggregated_params[key] = np.vectorize(
                lambda x: decode(self.paillier_private_key.decrypt(x), self.modulus)
            )(aggregated_params[key])
            aggregated_params[key] = aggregated_params[key] / DIVISIBLE_NUM
            aggregated_params[key] = re_integerize(aggregated_params[key], PRECISION)
            aggregated_params[key] = aggregated_params[key] / n_avg

        return aggregated_params

    def aggregate(self, silo_id_list_in_this_round: List[int], round_idx: int):
        """
        Aggregate the local trained models from the selected silos.

        Param:
            silo_id_list_in_this_round (list): the list of silo ids to be selected for each round of FL training.
        Return:
            averaged_params (dict): the averaged model parameters.
        """
        raw_client_model_or_grad_list = []
        for silo_id in silo_id_list_in_this_round:
            raw_client_model_or_grad_list.append(self.model_dict[silo_id])

        if self.strategy in ["ULDP-AVG-w"]:
            averaged_param_diff = self._aggregate_with_decrypt(
                raw_client_model_or_grad_list, self.n_users * self.n_silo_per_round
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy in ["ULDP-SGD-w"]:
            averaged_grads = self._aggregate_with_decrypt(
                raw_client_model_or_grad_list, self.n_users * self.n_silo_per_round
            )
            global_weights = self.update_parameters_from_gradients(
                averaged_grads, self.global_learning_rate
            )
        elif self.strategy in ["ULDP-AVG-ws"]:
            averaged_param_diff = self._aggregate_with_decrypt(
                raw_client_model_or_grad_list,
                self.n_users * self.n_silo_per_round * self.sampling_rate_q,
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy in ["ULDP-SGD-ws"]:
            averaged_grads = self._aggregate_with_decrypt(
                raw_client_model_or_grad_list,
                self.n_users * self.n_silo_per_round * self.sampling_rate_q,
            )
            global_weights = self.update_parameters_from_gradients(
                averaged_grads, self.global_learning_rate
            )
        else:
            raise NotImplementedError(
                "strategy = {} is not implemented for secure weighted version".format(
                    self.strategy
                )
            )

        self.record_epsilon(round_idx)
        return global_weights


class SecureLocalTrainer(ClassificationTrainer):
    """Implement Secure Aggregation.
    In Cross-silo FL, dropouts are not considered, so the mask is simpler.
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
        n_users: int,
        n_silos: int,
        client_optimizer: str = "sgd",
        local_learning_rate: float = 0.001,
        local_batch_size: int = 1,
        weight_decay: float = 0.001,
        local_epochs: int = 5,
        local_delta: Optional[float] = None,
        local_sigma: Optional[float] = None,
        local_clipping_bound: Optional[float] = None,
        group_k: Optional[int] = None,
        user_weights: Optional[Dict[int, float]] = None,
        n_silo_per_round: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ):
        super().__init__(
            base_seed,
            model,
            silo_id,
            device,
            agg_strategy,
            local_train_dataset,
            local_test_dataset,
            user_histogram,
            user_ids_of_local_train_dataset,
            client_optimizer,
            local_learning_rate,
            local_batch_size,
            weight_decay,
            local_epochs,
            local_delta,
            local_sigma,
            local_clipping_bound,
            group_k,
            user_weights,
            n_silo_per_round,
            dataset_name,
        )
        self.dh_seckey = pyDH.DiffieHellman(group=15)  # 3072-bit security
        self.dh_pubkey = self.dh_seckey.gen_public_key()
        self.n_users = n_users
        self.n_silos = n_silos
        self.multiplicative_masks = dict()
        for user_id in range(n_users):
            if user_id not in self.user_histogram:
                self.user_histogram[user_id] = 0

    def get_dh_pubkey(self) -> int:
        return self.dh_pubkey

    def set_paillier_public_key(self, paillier_public_key: phe.PaillierPublicKey):
        self.paillier_public_key = paillier_public_key
        self.modulus = paillier_public_key.n

    def gen_shared_keys(self, shared_dh_pubkey_dict: Dict[int, int]):
        shared_keys_per_silo = dict()
        for silo_id, shared_dh_pubkey in shared_dh_pubkey_dict.items():
            shared_keys_per_silo[silo_id] = int(
                self.dh_seckey.gen_shared_key(shared_dh_pubkey), 16
            )
        self.shared_keys_per_silo = shared_keys_per_silo

    def secagg_mask_params(self, encrypted_model_delta: OrderedDict, round_idx: int):
        if len(self.shared_keys_per_silo) < self.n_silo_per_round:
            raise Exception("Not all shared keys are received")
        masked_encrypted_model_delta = copy.deepcopy(encrypted_model_delta)
        for silo_id, shared_key in self.shared_keys_per_silo.items():
            for name, weight in encrypted_model_delta.items():
                masks = self.gen_random_additive_GFp_masks_by_pairwise_shared_key(
                    weight.shape, shared_key, round_idx=round_idx, suffix=name
                )
                if self.silo_id < silo_id:
                    masked_encrypted_model_delta[name] += masks
                elif self.silo_id > silo_id:
                    masked_encrypted_model_delta[name] -= masks
        return masked_encrypted_model_delta

    def secagg_mask_user_hist(self, user_hist: Dict[int, int]) -> Dict[int, int]:
        if len(self.shared_keys_per_silo) < self.n_silo_per_round:
            raise Exception("Not all shared keys are received")
        masked_user_hist = copy.deepcopy(user_hist)
        for silo_id, shared_key in self.shared_keys_per_silo.items():
            masks = self.gen_random_additive_GFp_masks_by_pairwise_shared_key(
                (self.n_users,), shared_key, suffix="user_hist"
            )
            for user_id in masked_user_hist.keys():
                if self.silo_id < silo_id:
                    masked_user_hist[user_id] += masks[user_id]
                elif self.silo_id > silo_id:
                    masked_user_hist[user_id] -= masks[user_id]
                masked_user_hist[user_id] = masked_user_hist[user_id] % self.modulus
        return masked_user_hist

    def gen_random_additive_GFp_masks_by_pairwise_shared_key(
        self, shape: Tuple, shared_key: int, round_idx: int = 0, suffix: str = ""
    ):
        combined = str(round_idx) + str(shared_key) + suffix
        hashed = hashlib.sha256(combined.encode()).hexdigest()
        seed = int(hashed, 16)
        if seed.bit_length() > 32:
            seed = seed & ((1 << 32) - 1)
        random_state = np.random.RandomState(seed)
        random_array = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            random_array[idx] = random_in_GFp(
                self.paillier_public_key.max_int, random_state
            )
        return random_array

    def gen_across_silos_shared_random_seed(self) -> Dict[int, int]:
        if self.silo_id != PRIMARY_SILO_ID:
            raise Exception("Only primary silo can generate shared random seed")
        self.across_silos_shared_random_seed = self.random_state.randint(0, 2**32 - 1)
        encrypted_random_seed_for_each_silo = dict()
        for silo_id in range(self.n_silo_per_round):
            if silo_id != PRIMARY_SILO_ID:
                encrypted_random_seed_for_each_silo[silo_id] = (
                    self.across_silos_shared_random_seed
                    ^ self.shared_keys_per_silo[silo_id]
                )
        return encrypted_random_seed_for_each_silo

    def receive_across_silos_shared_random_seed(self, shared_random_seed: int):
        self.across_silos_shared_random_seed = (
            shared_random_seed ^ self.shared_keys_per_silo[PRIMARY_SILO_ID]
        )

    def gen_random_multiplicative_masks_for_each_user_by_across_silos_shared_random_seed(
        self,
    ):
        if self.across_silos_shared_random_seed is None:
            raise Exception("Shared random seed is not received")
        # using the same random state for all silos
        multiplicative_random_masks_random_state = np.random.RandomState(
            self.across_silos_shared_random_seed
        )
        for user_id in range(self.n_users):
            # Generate a random number on the n^2 residue class ring of the Paillier public key
            # In most cases, there exists the inverse element on the the ring
            # https://crypto.stackexchange.com/questions/5636/inverse-element-in-paillier-cryptosystem
            # r^n needs to be coprime with n^2 for existing multiplicative inverse in Z_{n^2}
            # Necessary condition is gcd(r,n)=1
            self.multiplicative_masks[user_id] = random_in_GFp(
                self.paillier_public_key.max_int,
                multiplicative_random_masks_random_state,
            )
            assert (
                math.gcd(self.multiplicative_masks[user_id], self.paillier_public_key.n)
                == 1
            ), "r and n are not coprime, multiplicative inverse does not exist!"

    def multiplicative_blind_user_hist(self):
        multiplicative_blinded_user_hist = dict()
        for user_id, count in self.user_histogram.items():
            multiplicative_blinded_user_hist[user_id] = (
                count * self.multiplicative_masks[user_id]
            ) % self.modulus
        return multiplicative_blinded_user_hist

    def receive_encrypted_weights(
        self, encrypted_weights: Dict[int, phe.EncryptedNumber]
    ):
        self.encrypted_weights: Dict[int, phe.EncryptedNumber] = encrypted_weights

    def secure_weighting(self, model_delta: OrderedDict, user_id: int) -> OrderedDict:
        # take the model delta as input, and return the encrypted model delta
        n = self.user_histogram[user_id]
        r = self.multiplicative_masks[user_id]
        enc = self.encrypted_weights[user_id]
        encrypted_model_delta = OrderedDict()
        for key, value in model_delta.items():
            int_param = integerize(value.numpy(), PRECISION)
            encoded_int_param = encode(int_param, self.modulus)
            encrypted_model_delta[key] = enc * n
            encrypted_model_delta[key] *= r
            encrypted_model_delta[key] *= DIVISIBLE_NUM
            # vector computataion through numpy
            encrypted_model_delta[key] = encoded_int_param * encrypted_model_delta[key]
        return encrypted_model_delta

    def secure_add_noise(
        self, sigma: float, encrypted_model_delta: OrderedDict
    ) -> OrderedDict:
        noised_encrypted_model_delta = OrderedDict()
        for key, value in encrypted_model_delta.items():
            noise = self.random_state.normal(0, sigma, value.shape)
            int_noise = integerize(noise.numpy(), PRECISION)
            encoded_int_noise = encode(int_noise, self.modulus)
            noised_encrypted_model_delta[key] = (
                encoded_int_noise * DIVISIBLE_NUM + value
            )
        return noised_encrypted_model_delta
