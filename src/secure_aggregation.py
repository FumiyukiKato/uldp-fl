import copy
import math
import time
from typing import Callable, Dict, List, Optional, OrderedDict, Tuple
import numpy as np
from phe import paillier
import phe
import torch
import pyDH
import hashlib

from aggregator import Aggregator
from local_trainer import ClassificationTrainer
from method_group import (
    METHOD_GROUP_AVG,
    METHOD_ULDP_AVG_W,
    METHOD_ULDP_AVG_WS,
    METHOD_ULDP_SGD_W,
    METHOD_ULDP_SGD_WS,
    METHOD_GROUP_GRADIENT,
    METHOD_GROUP_WEIGHTS,
)
import noise_utils
from mylogger import logger

"""
This file implements the private weighting protocol.
"""

N_LENGTH = 3072
PRECISION = 1e-10  # considering python's float64 precision, ~ 1e-15 is the appropriate precision
MAX_N_USER = 2000
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


def gen_random_int_in_GFp(p: int, random_state: np.random.RandomState):
    n_length = p.bit_length()
    while True:
        bits = random_state.randint(0, 2, n_length)
        random_number = int("".join(map(str, bits)), 2)
        if random_number < p:
            return random_number


def gen_random_GFp_masks(p: int, shape: Tuple, random_state: np.random.RandomState):
    random_array = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        random_array[idx] = gen_random_int_in_GFp(p, random_state)
    return random_array


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


def build_random_seed(shared_key: int = 0, round_idx: int = 0, suffix: str = "") -> int:
    combined = str(round_idx) + str(shared_key) + suffix
    hashed = hashlib.sha256(combined.encode()).hexdigest()
    seed = int(hashed, 16)
    if seed.bit_length() > 32:
        seed = seed & ((1 << 32) - 1)
    return seed


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
        self.base_seed = base_seed
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
        logger.debug("SERVER: Paillier key generated")

    def receive_dh_pubkey(
        self, silo_id: int, sent_dh_pubkey: int
    ) -> Optional[Dict[int, int]]:
        self.client_keys[silo_id] = sent_dh_pubkey
        self.dh_exchange_ready_client_ids.add(silo_id)
        if len(self.dh_exchange_ready_client_ids) == self.n_silo_per_round:
            return self.client_keys

    def receive_blinded_user_histogram(
        self, silo_id: int, sent_blinded_user_records_histogram: Dict[int, int]
    ) -> Optional[Dict[int, int]]:
        for user_id, blinded_count in sent_blinded_user_records_histogram.items():
            self.blinded_user_histogram[user_id] += blinded_count
            self.blinded_user_histogram[user_id] %= self.modulus
        self.histogram_ready_client_ids.add(silo_id)
        if len(self.histogram_ready_client_ids) == self.n_silo_per_round:
            self.inversed_blinded_user_histogram = self._compute_inverse(
                self.blinded_user_histogram, self.modulus
            )
            return self.inversed_blinded_user_histogram

    def receive_shared_random_seed(
        self, sent_shared_random_seed_per_silo: Dict[int, int]
    ):
        self.shared_random_seed_per_silo = sent_shared_random_seed_per_silo

    def get_shared_random_seed(self) -> Dict[int, int]:
        return self.shared_random_seed_per_silo

    def _compute_inverse(
        self, blinded_user_histogram: Dict[int, int], modulus: int
    ) -> Dict[int, int]:
        inversed_blinded_user_histogram = dict()
        for user_id, blinded_count in blinded_user_histogram.items():
            inversed_blinded_user_histogram[user_id] = non_recursive_mod_inverse(
                blinded_count, modulus
            )
        return inversed_blinded_user_histogram

    def get_encrypt_inversed_blinded_user_histogram_with_userlevel_subsampling(
        self,
        round_idx: int,
    ) -> Dict[int, int]:
        sampled_inversed_blinded_user_histogram = copy.deepcopy(
            self.inversed_blinded_user_histogram
        )
        user_ids = np.array(range(self.n_users))
        sampled_user_ids = user_ids[
            np.random.RandomState(
                seed=build_random_seed(
                    self.base_seed, round_idx=round_idx, suffix="subsampling"
                )
            ).rand(len(user_ids))
            < self.sampling_rate_q
        ]
        sampled_user_ids_set = set(sampled_user_ids)
        for user_id in range(self.n_users):
            if user_id not in sampled_user_ids_set:
                sampled_inversed_blinded_user_histogram[user_id] = 0
        return self._encrypt_histogram(sampled_inversed_blinded_user_histogram)

    def get_encrypt_inversed_blinded_user_histogram(
        self,
    ) -> Dict[int, phe.EncryptedNumber]:
        return self._encrypt_histogram(self.inversed_blinded_user_histogram)

    def _encrypt_histogram(
        self, histogram: Dict[int, int]
    ) -> Dict[int, phe.EncryptedNumber]:
        encrypted = dict()
        for user_id, value in histogram.items():
            encrypted[user_id] = self.paillier_public_key.encrypt(value)
        return encrypted

    def add_local_trained_result(self, silo_id, model_params, eps):
        self.model_dict[silo_id] = model_params
        self.flag_client_model_uploaded_dict[silo_id] = True
        self.latest_eps = max(self.latest_eps, eps)

    def _aggregate_with_decrypt(
        self, param_list: List[OrderedDict], n_avg: float
    ) -> OrderedDict[str, torch.Tensor]:
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
            aggregated_params[key] = torch.from_numpy(aggregated_params[key])

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

        if self.strategy == METHOD_ULDP_AVG_W:
            averaged_param_diff = self._aggregate_with_decrypt(
                raw_client_model_or_grad_list, self.n_users * self.n_silo_per_round
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy == METHOD_ULDP_SGD_W:
            averaged_grads = self._aggregate_with_decrypt(
                raw_client_model_or_grad_list, self.n_users * self.n_silo_per_round
            )
            global_weights = self.update_parameters_from_gradients(
                averaged_grads, self.global_learning_rate
            )
        elif self.strategy == METHOD_ULDP_AVG_WS:
            averaged_param_diff = self._aggregate_with_decrypt(
                raw_client_model_or_grad_list,
                self.n_users * self.n_silo_per_round * self.sampling_rate_q,
            )
            global_weights = self.update_global_weights_from_diff(
                averaged_param_diff, self.global_learning_rate
            )
        elif self.strategy == METHOD_ULDP_SGD_WS:
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
        self.base_seed = base_seed
        self.dh_seckey = pyDH.DiffieHellman(group=15)  # 3072-bit security
        self.dh_pubkey = self.dh_seckey.gen_public_key()
        logger.debug("SILO: DH key generated")
        self.n_users = n_users
        self.n_silos = n_silos
        self.multiplicative_masks = dict()
        for user_id in range(n_users):
            if user_id not in self.user_histogram:
                self.user_histogram[user_id] = 0

    def get_dh_pubkey(self) -> int:
        return self.dh_pubkey

    def receive_paillier_public_key(self, paillier_public_key: phe.PaillierPublicKey):
        self.paillier_public_key = paillier_public_key
        self.modulus = paillier_public_key.n

    def receive_dh_pubkeys_and_gen_shared_keys(
        self, shared_dh_pubkey_dict: Dict[int, int]
    ):
        shared_keys_per_silo = dict()
        for silo_id, shared_dh_pubkey in shared_dh_pubkey_dict.items():
            shared_keys_per_silo[silo_id] = int(
                self.dh_seckey.gen_shared_key(shared_dh_pubkey), 16
            )
        self.shared_keys_per_silo: Dict[int, int] = shared_keys_per_silo

    def secagg_mask_params(
        self, encrypted_model_delta: OrderedDict, round_idx: int, modulus: int
    ):
        masked_encrypted_model_delta = copy.deepcopy(encrypted_model_delta)
        for silo_id, shared_key in self.shared_keys_per_silo.items():
            for name, weight in encrypted_model_delta.items():
                seed = build_random_seed(shared_key, round_idx, suffix=name)
                masks = gen_random_GFp_masks(
                    modulus, weight.shape, np.random.RandomState(seed)
                )
                if self.silo_id < silo_id:
                    masked_encrypted_model_delta[name] += masks
                elif self.silo_id > silo_id:
                    masked_encrypted_model_delta[name] -= masks
        return masked_encrypted_model_delta

    def _additive_mask_for_secagg_user_hist(
        self,
        user_hist: Dict[int, int],
        shared_keys_per_silo: Dict[int, int],
        n_users: int,
        self_silo_id: int,
        modulus: int,
    ) -> Dict[int, int]:
        masked_user_hist = copy.deepcopy(user_hist)
        for silo_id, shared_key in shared_keys_per_silo.items():
            seed = build_random_seed(shared_key, suffix="user_hist")
            masks = gen_random_GFp_masks(
                modulus, (n_users,), np.random.RandomState(seed)
            )
            for user_id in masked_user_hist.keys():
                if self_silo_id < silo_id:
                    masked_user_hist[user_id] += masks[user_id]
                elif self_silo_id > silo_id:
                    masked_user_hist[user_id] -= masks[user_id]
                masked_user_hist[user_id] = masked_user_hist[user_id] % modulus
        return masked_user_hist

    def gen_across_silos_shared_random_seed(self) -> Dict[int, int]:
        if self.silo_id != PRIMARY_SILO_ID:
            raise Exception("Only primary silo can generate shared random seed")
        self.across_silos_shared_random_seed = np.random.RandomState(
            seed=build_random_seed(
                shared_key=self.base_seed, suffix="across_silos_shared_random_seed"
            )
        ).randint(0, 2**32 - 1)
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

    def make_blinded_user_hist(self) -> Dict[int, int]:
        self.multiplicative_masks = self._gen_random_multiplicative_masks(
            self.across_silos_shared_random_seed
        )
        multiplicative_blinded_user_hist = self._multiplicative_blind_user_hist(
            self.multiplicative_masks, self.user_histogram, self.modulus
        )
        if len(self.shared_keys_per_silo) < self.n_silo_per_round:
            raise Exception("Not all shared keys are received")
        masked_user_hist = self._additive_mask_for_secagg_user_hist(
            multiplicative_blinded_user_hist,
            self.shared_keys_per_silo,
            self.n_users,
            self.silo_id,
            self.modulus,
        )
        return masked_user_hist

    def _gen_random_multiplicative_masks(
        self, across_silos_shared_random_seed: int
    ) -> Dict[int, int]:
        # using the same random state for all silos
        multiplicative_random_masks_random_state = np.random.RandomState(
            across_silos_shared_random_seed
        )
        multiplicative_masks = dict()
        for user_id in range(self.n_users):
            # Generate a random number on the n residue class ring of the Paillier public key
            # In most cases, there exists the inverse element on the the ring
            # Necessary condition is gcd(r,n)=1
            # https://crypto.stackexchange.com/questions/5636/inverse-element-in-paillier-cryptosystem
            multiplicative_masks[user_id] = gen_random_int_in_GFp(
                self.paillier_public_key.max_int,
                multiplicative_random_masks_random_state,
            )
            assert (
                math.gcd(multiplicative_masks[user_id], self.paillier_public_key.n) == 1
            ), "r and n are not coprime, multiplicative inverse does not exist!"
        return multiplicative_masks

    def _multiplicative_blind_user_hist(
        self,
        multiplicative_masks: Dict[int, int],
        user_histogram: Dict[int, int],
        modulus: int,
    ) -> Dict[int, int]:
        multiplicative_blinded_user_hist = dict()
        for user_id, count in user_histogram.items():
            multiplicative_blinded_user_hist[user_id] = (
                count * multiplicative_masks[user_id]
            ) % modulus
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

    def _sum_params(self, param_list: List[OrderedDict]) -> OrderedDict:
        summed_params = copy.deepcopy(param_list[0])
        for k in summed_params.keys():
            for i in range(1, len(param_list)):
                summed_params[k] += param_list[i][k]
        return summed_params

    def _secure_add_noise(
        self,
        encrypted_model_delta: OrderedDict,
        sigma: float,
        random_state: np.random.RandomState,
        modulus: int,
    ) -> OrderedDict:
        noised_encrypted_model_delta = OrderedDict()
        for key, value in encrypted_model_delta.items():
            noise = random_state.normal(0, sigma, value.shape)
            int_noise = integerize(noise, PRECISION)
            encoded_int_noise = encode(int_noise, modulus)
            noised_encrypted_model_delta[key] = (
                encoded_int_noise * DIVISIBLE_NUM
            ) % modulus + value
        return noised_encrypted_model_delta

    def train(
        self, global_round_index: int, loss_callback: Callable = lambda loss: None
    ) -> OrderedDict:
        """
        Train the model on the local dataset.
        """
        tick = time.time()

        model = self.model
        model.to(self.device)
        model.train()
        global_weights = copy.deepcopy(self.get_model_params())

        torch.manual_seed(self.get_torch_manual_seed())

        criterion = self.criterion

        if self.agg_strategy in METHOD_GROUP_GRADIENT.intersection(
            METHOD_GROUP_WEIGHTS
        ):
            grads_list = []  # TODO: memory optimization (use online aggregation)
            for user_id, user_train_loader in self.user_level_data_loader:
                logger.debug("User %d" % user_id)
                user_avg_grad = OrderedDict()
                for name, param in model.named_parameters():
                    user_avg_grad[name] = torch.zeros_like(param.data)

                for x, labels in user_train_loader:
                    x, labels = x.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    log_probs = model(x)
                    if self.dataset_name in ["creditcard"]:
                        labels = labels.long()
                    loss = criterion(log_probs, labels)
                    loss_callback(loss)
                    loss.backward()
                    # Don't optimize (i.e., Don't call step())

                    for name, param in model.named_parameters():
                        # Due to different batch size for each user
                        user_avg_grad[name] += param.grad / len(x)

                clipped_grads = noise_utils.global_clip(
                    model, user_avg_grad, self.local_clipping_bound
                )
                weighted_clipped_grads = self.secure_weighting(clipped_grads, user_id)
                grads_list.append(weighted_clipped_grads)

            # calculate the average gradient
            avg_grads = self._sum_params(grads_list)
            noisy_avg_grads = self._secure_add_noise(
                avg_grads,
                sigma=self.local_sigma
                * self.local_clipping_bound
                / np.sqrt(self.n_silo_per_round),
                random_state=self.random_state,
                modulus=self.modulus,
            )
            masked_grad = self.secagg_mask_params(
                noisy_avg_grads,
                round_idx=global_round_index,
                modulus=self.modulus,
            )

        elif self.agg_strategy in METHOD_GROUP_AVG.intersection(METHOD_GROUP_WEIGHTS):

            def loss_callback(loss):
                if torch.isnan(loss):
                    logger.warn("loss is nan: skipping")
                    return True
                return False

            weights_diff_list = []  # TODO: memory optimization (use online aggregation)
            for user_id, user_train_loader in self.user_level_data_loader:
                logger.debug("User %d" % user_id)
                model_u = copy.deepcopy(model)
                # optimizer_u = torch.optim.SGD(
                #     filter(lambda p: p.requires_grad, model_u.parameters()),
                #     lr=self.local_learning_rate,
                # )
                optimizer_u = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model_u.parameters()),
                    lr=self.local_learning_rate,
                )

                for epoch in range(self.local_epochs):
                    batch_loss = []
                    for x, labels in user_train_loader:
                        x, labels = x.to(self.device), labels.to(self.device)
                        optimizer_u.zero_grad()
                        log_probs = model_u(x)
                        if self.dataset_name in ["creditcard"]:
                            labels = labels.long()
                        loss = criterion(log_probs, labels)
                        if loss_callback(loss):
                            continue
                        loss.backward()
                        optimizer_u.step()
                        batch_loss.append(loss.item())

                weights = model_u.state_dict()
                weights_diff = noise_utils.diff_weights(global_weights, weights)
                clipped_weights_diff = noise_utils.global_clip(
                    model_u, weights_diff, self.local_clipping_bound
                )
                weighted_clipped_weights_diff = self.secure_weighting(
                    clipped_weights_diff, user_id
                )
                weights_diff_list.append(weighted_clipped_weights_diff)

            avg_weights_diff = self._sum_params(weights_diff_list)
            noisy_avg_weights_diff = self._secure_add_noise(
                avg_weights_diff,
                self.local_sigma
                * self.local_clipping_bound
                / np.sqrt(self.n_silo_per_round),
                random_state=self.random_state,
                modulus=self.modulus,
            )
            masked_diff = self.secagg_mask_params(
                noisy_avg_weights_diff,
                round_idx=global_round_index,
                modulus=self.modulus,
            )

        else:
            raise NotImplementedError(
                "Unknown aggregation strategy for secure weighting"
            )

        train_time = time.time() - tick
        logger.debug("Train/Time : %s", train_time)
        self.results["train_time"].append((global_round_index, train_time))

        if self.agg_strategy in METHOD_GROUP_GRADIENT.intersection(
            METHOD_GROUP_WEIGHTS
        ):
            return masked_grad
        elif self.agg_strategy in METHOD_GROUP_AVG.intersection(METHOD_GROUP_WEIGHTS):
            return masked_diff
        else:
            raise NotImplementedError("Unknown aggregation strategy")
