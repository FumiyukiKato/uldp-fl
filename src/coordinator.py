from typing import Dict, Optional, List, Tuple
import numpy as np
import copy
from opacus.accountants import RDPAccountant
from method_group import (
    METHOD_GROUP_ONLINE_OPTIMIZATION,
    METHOD_PULDP_AVG,
    METHOD_ULDP_GROUP_MAX,
    METHOD_ULDP_GROUP_MEDIAN,
)

from mylogger import logger
import noise_utils
import epsilon_allocate_utils


class Coordinator:
    """
    Coordinator class for federated learning.
    Receive user distribution from silos and decide weighting strategies on shared parameters for each silo.
    """

    def __init__(
        self,
        base_seed: int,
        n_silos: int,
        n_users: int,
        group_k: int = None,
        agg_strategy: str = None,
        sampling_rate_q: float = None,
        q_u: Optional[Dict] = None,
        epsilon_u: Optional[Dict] = None,
        group_thresholds: Optional[List] = None,
        delta: Optional[float] = None,
        sigma: Optional[float] = None,
        n_total_round: Optional[int] = None,
        q_step_size: Optional[float] = None,
        hp_baseline: Optional[str] = None,
        step_decay: Optional[bool] = False,
        initial_q_u: Optional[float] = None,
    ):
        self.random_state = np.random.RandomState(seed=base_seed + 2000000)
        self.n_users = n_users
        self.n_silos = n_silos
        self.group_k = group_k
        self.sampling_rate_q = sampling_rate_q
        self.agg_strategy = agg_strategy
        self.ready_silos = set()
        self.original_user_hist_dct = {silo_id: {} for silo_id in range(n_silos)}
        self.q_u = q_u
        self.hp_baseline = hp_baseline

        if self.agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
            assert (
                q_step_size is not None and delta is not None and sigma is not None
            ), "q_step_size, delta, sigma must be given for Online optimization"
            self.q_step_size = q_step_size
            self.delta, self.sigma, self.n_total_round = delta, sigma, n_total_round
            self.step_decay = step_decay

            if self.hp_baseline:
                # To total eps (model optimization and HP optimization) is upper bounded
                self.n_release = self.n_total_round
            else:
                self.n_release = self.n_total_round * 3
            self.epsilon_groups = epsilon_allocate_utils.group_by_closest_below(
                epsilon_u_dct=epsilon_u, group_thresholds=group_thresholds
            )
            self.hp_dct_by_eps: Dict[float, Tuple] = {}
            self.param_history: Dict[float, List] = {}
            self.loss_history: Dict[float, List] = {}

            INITIAL_Q_U = 1.0
            if initial_q_u is not None:
                INITIAL_Q_U = initial_q_u
            for eps_u in self.epsilon_groups.keys():
                initial_C_u, _, _ = noise_utils.from_q_u(
                    q_u=INITIAL_Q_U,
                    delta=self.delta,
                    epsilon_u=eps_u,
                    sigma=self.sigma,
                    T=self.n_release,
                )
                self.hp_dct_by_eps[eps_u] = (INITIAL_Q_U, initial_C_u)
                self.param_history[eps_u] = [(INITIAL_Q_U, initial_C_u)]
                self.loss_history[eps_u] = []

    def set_user_hist_by_silo_id(self, silo_id: int, user_hist: Dict):
        self.ready_silos.add(silo_id)
        self.original_user_hist_dct[silo_id] = user_hist

    def is_ready(self) -> bool:
        if len(self.ready_silos) == self.n_silos:
            if self.agg_strategy == METHOD_ULDP_GROUP_MAX:
                group_max = self.get_group_max()
                logger.info(f"Group max: {group_max}")
                self.set_group_k(group_max)
            elif self.agg_strategy == METHOD_ULDP_GROUP_MEDIAN:
                group_median = self.get_group_median()
                logger.info(f"Group median: {group_median}")
                self.set_group_k(group_median)
            return True
        return False

    def set_group_k(self, group_k: int):
        self.group_k = group_k

    def get_epsilon_groups(self):
        return self.epsilon_groups

    def get_group_max(self):
        """
        Get the maximum number of users in a silo.
        """
        total_user_count = {}
        for silo_id, user_hist in self.original_user_hist_dct.items():
            for user_id, user_count in user_hist.items():
                if user_id not in total_user_count:
                    total_user_count[user_id] = 0
                total_user_count[user_id] += user_count
        return max(total_user_count.values())

    def get_group_median(self):
        """
        Get the median number of users in a silo.
        """
        total_user_count = {}
        for silo_id, user_hist in self.original_user_hist_dct.items():
            for user_id, user_count in user_hist.items():
                if user_id not in total_user_count:
                    total_user_count[user_id] = 0
                total_user_count[user_id] += user_count
        return int(np.median(list(total_user_count.values())))

    def build_user_weights(
        self,
        weighted: bool = False,
        is_sample: bool = False,
    ) -> Dict[int, Dict[int, float]]:
        """
        Build user weights for ULDP-SGD/AVG.
        """
        user_weights_per_silo = {
            silo_id: {user_id: 0.0 for user_id in range(self.n_users)}
            for silo_id in range(self.n_silos)
        }

        if weighted:
            # Weighting in proportion to the number of users
            # calculate total user count for each user
            total_user_count = {}
            for silo_id, user_hist in self.original_user_hist_dct.items():
                for user_id, user_count in user_hist.items():
                    if user_id not in total_user_count:
                        total_user_count[user_id] = 0
                    total_user_count[user_id] += user_count
            # calculate user weights for each silo
            for silo_id, user_hist in self.original_user_hist_dct.items():
                for user_id, user_count in user_hist.items():
                    user_weights_per_silo[silo_id][user_id] = (
                        user_count / total_user_count[user_id]
                    )
        else:
            for silo_id in range(self.n_silos):
                user_weights_per_silo[silo_id] = {
                    user_id: 1.0 / self.n_silos for user_id in range(self.n_users)
                }
        if is_sample:
            user_ids = np.array(range(self.n_users))
            if self.agg_strategy == METHOD_PULDP_AVG:
                sampled_user_ids = user_ids[
                    self.random_state.rand(len(self.q_u))
                    < np.array(list(self.q_u.values()))
                ]
            else:
                sampled_user_ids = user_ids[
                    self.random_state.rand(len(user_ids)) < self.sampling_rate_q
                ]
            sampled_user_ids_set = set(sampled_user_ids)
            for silo_id in range(self.n_silos):
                for user_id in range(self.n_users):
                    if user_id not in sampled_user_ids_set:
                        user_weights_per_silo[silo_id][user_id] = 0.0

        return user_weights_per_silo

    def build_user_weights_with_online_optimization(
        self,
        weighted: bool = False,
        round_idx: Optional[int] = None,
        accountant_dct: Optional[Dict[float, RDPAccountant]] = None,
    ) -> Tuple[
        Dict[int, Dict[int, float]],
        np.ndarray,
        Dict[int, Dict[int, float]],
        np.ndarray,
        Dict[int, Dict[int, float]],
        np.ndarray,
    ]:
        user_weights_per_silo = {
            silo_id: {user_id: 0.0 for user_id in range(self.n_users)}
            for silo_id in range(self.n_silos)
        }

        if weighted:
            # Weighting in proportion to the number of users
            # calculate total user count for each user
            total_user_count = {}
            for silo_id, user_hist in self.original_user_hist_dct.items():
                for user_id, user_count in user_hist.items():
                    if user_id not in total_user_count:
                        total_user_count[user_id] = 0
                    total_user_count[user_id] += user_count
            # calculate user weights for each silo
            for silo_id, user_hist in self.original_user_hist_dct.items():
                for user_id, user_count in user_hist.items():
                    user_weights_per_silo[silo_id][user_id] = (
                        user_count / total_user_count[user_id]
                    )
        else:
            for silo_id in range(self.n_silos):
                user_weights_per_silo[silo_id] = {
                    user_id: 1.0 / self.n_silos for user_id in range(self.n_users)
                }

        user_weights_per_silo_for_optimization = copy.deepcopy(user_weights_per_silo)
        stepped_user_weights_per_silo_for_optimization = copy.deepcopy(
            user_weights_per_silo
        )

        user_ids = np.array(range(self.n_users))

        self.q_u_list = np.zeros(self.n_users)
        self.C_u_list = np.zeros(self.n_users)
        self.stepped_q_u_list = np.zeros(self.n_users)
        self.stepped_C_u_list = np.zeros(self.n_users)
        q_step_size = schedule_step_size(
            self.q_step_size, round_idx, self.n_total_round, step_decay=self.step_decay
        )
        for eps_u, user_ids_per_eps in self.epsilon_groups.items():
            q_u, C_u = self.hp_dct_by_eps[eps_u]
            self.q_u_list[user_ids_per_eps] = q_u
            self.C_u_list[user_ids_per_eps] = C_u

            if self.hp_baseline is None:
                accountant = copy.deepcopy(accountant_dct[eps_u])
                # consume DP for 2 times publish to accurately calculate the next q_u which is not over the given epsilon constraint
                accountant.step(noise_multiplier=self.sigma / C_u, sample_rate=q_u)
                accountant.step(noise_multiplier=self.sigma / C_u, sample_rate=q_u)
                step_size = q_step_size
                stepped_q_u, stepped_C_u = compute_stepped_qC(
                    step_size=step_size,
                    q_u=q_u,
                    delta=self.delta,
                    eps_u=eps_u,
                    sigma=self.sigma,
                    total_round=self.n_release,
                    current_round=round_idx * 3
                    + 2,  # 3 times per round + original q_u of this round
                    current_accountant=accountant,
                )
                self.stepped_q_u_list[user_ids_per_eps] = stepped_q_u
                self.stepped_C_u_list[user_ids_per_eps] = stepped_C_u

        sampled_user_ids = user_ids[
            self.random_state.rand(len(self.q_u_list)) < self.q_u_list
        ]
        sampled_user_ids_set = set(sampled_user_ids)

        sampled_user_ids_for_optimization = user_ids[
            self.random_state.rand(len(self.q_u_list)) < self.q_u_list
        ]
        sampled_user_ids_set_for_optimization = set(sampled_user_ids_for_optimization)

        stepped_sampled_user_ids_for_optimization = user_ids[
            self.random_state.rand(len(self.stepped_q_u_list)) < self.stepped_q_u_list
        ]
        stepped_sampled_user_ids_set_for_optimization = set(
            stepped_sampled_user_ids_for_optimization
        )
        for silo_id in range(self.n_silos):
            for user_id in range(self.n_users):
                if user_id not in sampled_user_ids_set:
                    user_weights_per_silo[silo_id][user_id] = 0.0
                if user_id not in sampled_user_ids_set_for_optimization:
                    user_weights_per_silo_for_optimization[silo_id][user_id] = 0.0
                if user_id not in stepped_sampled_user_ids_set_for_optimization:
                    stepped_user_weights_per_silo_for_optimization[silo_id][
                        user_id
                    ] = 0.0

        return (
            user_weights_per_silo,
            self.C_u_list,
            user_weights_per_silo_for_optimization,
            self.C_u_list,
            stepped_user_weights_per_silo_for_optimization,
            self.stepped_C_u_list,
        )

    def build_user_bound_histograms(
        self,
        old_user_histogram_dct: Optional[Dict[int, Dict[int, int]]] = None,
        minimum_number_of_records: int = 1,
    ) -> Dict[int, Dict[int, int]]:
        """
        Build user bounded histograms for ULDP-GROUP.
        """
        if old_user_histogram_dct is not None:
            # Fair method, if old_user_histogram_dct is given
            total_user_histogram = {}
            for _, user_histogram in old_user_histogram_dct.items():
                for user_id, user_count in user_histogram.items():
                    if user_id not in total_user_histogram:
                        total_user_histogram[user_id] = 0
                    total_user_histogram[user_id] += user_count

            # Assign to silos in a round-robin so that the sum of
            # each user's records is less than or equal to group_k
            round_robin_silo_idx = 0
            new_user_histogram_dct = {
                silo_id: {} for silo_id in old_user_histogram_dct.keys()
            }
            for user_id, user_count in total_user_histogram.items():
                current_user_count = 0
                while (
                    current_user_count < self.group_k
                    and current_user_count < user_count
                ):
                    if (
                        user_id in old_user_histogram_dct[round_robin_silo_idx]
                        and old_user_histogram_dct[round_robin_silo_idx][user_id] > 0
                    ):
                        old_user_histogram_dct[round_robin_silo_idx][
                            user_id
                        ] -= minimum_number_of_records
                        if user_id not in new_user_histogram_dct[round_robin_silo_idx]:
                            new_user_histogram_dct[round_robin_silo_idx][
                                user_id
                            ] = minimum_number_of_records
                        else:
                            new_user_histogram_dct[round_robin_silo_idx][
                                user_id
                            ] += minimum_number_of_records
                        current_user_count += minimum_number_of_records
                    round_robin_silo_idx = (round_robin_silo_idx + 1) % self.n_silos
            return new_user_histogram_dct

        else:
            # Randomly assign group_k capacity for users to each silo
            new_user_histogram_dct = {silo_id: {} for silo_id in range(self.n_silos)}
            count_matrix = self.random_state.choice(
                self.n_silos,
                (self.n_users, int(self.group_k / minimum_number_of_records)),
                replace=True,
            )
            for user_id in range(self.n_users):
                for silo_id in count_matrix[user_id]:
                    if user_id not in new_user_histogram_dct[silo_id]:
                        new_user_histogram_dct[silo_id][
                            user_id
                        ] = minimum_number_of_records
                    else:
                        new_user_histogram_dct[silo_id][
                            user_id
                        ] += minimum_number_of_records
            return new_user_histogram_dct

    def online_optimize(
        self,
        loss_diff_dct: Dict[float, float],
        with_momentum: bool = False,
        beta: float = 0.8,
        round_idx: Optional[int] = None,
        accountant_dct: Optional[Dict[float, RDPAccountant]] = None,
    ):
        q_step_size = schedule_step_size(
            self.q_step_size, round_idx, self.n_total_round, step_decay=self.step_decay
        )
        for eps_u, eps_user_ids in self.epsilon_groups.items():
            if self.hp_baseline is None or self.hp_baseline == "random-updown":
                if self.hp_baseline is None:
                    loss_diff = loss_diff_dct[
                        eps_u
                    ]  # diff = stepped_test_loss - test_loss
                    org_diff = loss_diff
                    if with_momentum:
                        if not hasattr(self, "momentum"):
                            self.momentum = {}
                            # self.data_memory = {}
                            # self.record_metrics = {}
                        if len(self.loss_history[eps_u]) == 0:
                            self.momentum[eps_u] = loss_diff_dct[eps_u]
                            # self.data_memory[eps_u] = [loss_diff]
                            # self.record_metrics[eps_u] = {
                            #     "original": [],
                            #     "clipped": [],
                            #     "ema": [],
                            # }
                        else:
                            loss_diff = (
                                beta * self.momentum[eps_u] + (1 - beta) * loss_diff
                            )
                            self.momentum[eps_u] = loss_diff

                            # # print("eps_u", eps_u)
                            # self.data_memory[eps_u].append(loss_diff)
                            # percentile = 50
                            # n_data_for_ema = 5
                            # data = self.data_memory[eps_u]
                            # clipping_bound = np.percentile(np.abs(data), percentile)
                            # clipped_data = np.clip(
                            #     data, -clipping_bound, clipping_bound
                            # )
                            # # print(
                            # #     f"original data: 移動平均5: {np.mean(data[-5:])}, 移動平均10: {np.mean(data[-10:])}, 移動平均20: {np.mean(data[-20:])}"
                            # # )
                            # self.record_metrics[eps_u]["original"].append(
                            #     (
                            #         np.mean(data[-5:]),
                            #         np.mean(data[-10:]),
                            #         np.mean(data[-20:]),
                            #     )
                            # )
                            # # print(
                            # #     f"clipped data: 移動平均5: {np.mean(clipped_data[-5:])}, 移動平均10: {np.mean(clipped_data[-10:])}, 移動平均20: {np.mean(clipped_data[-20:])}"
                            # # )
                            # self.record_metrics[eps_u]["clipped"].append(
                            #     (
                            #         np.mean(clipped_data[-5:]),
                            #         np.mean(clipped_data[-10:]),
                            #         np.mean(clipped_data[-20:]),
                            #     )
                            # )
                            # # print(
                            # #     f"clipped data: EMA 5: {calc_ema(clipped_data, beta, 5)}, EMA 10: {calc_ema(clipped_data, beta, 10)}, EMA 20: {calc_ema(clipped_data, beta, 20)}"
                            # # )
                            # self.record_metrics[eps_u]["ema"].append(
                            #     (
                            #         calc_ema(clipped_data, beta, 5),
                            #         calc_ema(clipped_data, beta, 10),
                            #         calc_ema(clipped_data, beta, 20),
                            #     )
                            # )
                            # ema = calc_ema(clipped_data, beta, n_data_for_ema)
                            # loss_diff = ema

                        logger.info(
                            f"eps_u: {eps_u}, original loss_diff: {org_diff}, (self.momentum[eps_u]: {self.momentum[eps_u]})"
                        )
                elif self.hp_baseline == "random-updown":
                    loss_diff = self.random_state.uniform(-1, 1)
                    org_diff = loss_diff

                q_u, _ = self.hp_dct_by_eps[eps_u]
                if loss_diff < 0:  # Model is getting better
                    q_u = q_u * q_step_size
                    q_u = max(q_u, 1.0 / len(eps_user_ids))
                else:  # Model is getting worse
                    q_u = q_u / q_step_size
                    q_u = min(q_u, 1.0)

            elif self.hp_baseline == "random":
                # Randomly select q_u between 1.0 and 1/n, so that it is evenly selected in the log scale
                random_value = self.random_state.uniform(1.0 / len(eps_user_ids), 1.0)
                q_u = random_value
                loss_diff = random_value
                org_diff = random_value
            elif self.hp_baseline == "random-log":
                # Randomly select q_u between 1.0 and 1/n, so that it is evenly selected in the log scale
                log_min = np.log10(1.0 / len(eps_user_ids))
                log_max = np.log10(1.0)
                random_log_value = self.random_state.uniform(log_min, log_max)
                q_u = 10**random_log_value
                loss_diff = random_log_value
                org_diff = random_log_value
            else:
                raise ValueError(f"hp_baseline {self.hp_baseline} is not supported.")

            # Next (q_u, C_u)
            if self.hp_baseline:
                current_round = round_idx + 1
            else:
                current_round = (round_idx + 1) * 3
            C_u, _ = noise_utils.from_q_u_with_history(
                q_u,
                self.delta,
                eps_u,
                self.sigma,
                total_round=self.n_release,
                current_round=current_round,
                current_accountant=accountant_dct[eps_u],
            )
            self.hp_dct_by_eps[eps_u] = (q_u, C_u)
            self.param_history[eps_u].append((q_u, C_u))
            self.loss_history[eps_u].append((loss_diff, org_diff))


def compute_stepped_qC(
    step_size: float,
    q_u: float,
    delta: float,
    eps_u: float,
    sigma: float,
    total_round: int,
    current_round: int,
    current_accountant: RDPAccountant = None,
):
    dst_q = q_u * (step_size)
    dst_C, _ = noise_utils.from_q_u_with_history(
        q_u=dst_q,
        delta=delta,
        epsilon_u=eps_u,
        sigma=sigma,
        total_round=total_round,
        current_round=current_round,
        current_accountant=current_accountant,
    )
    return dst_q, dst_C


def schedule_step_size(
    step_size: float, round_idx: int, n_total_round: int, step_decay=False
):
    if step_decay:
        return 1 - (1 - step_size) * (1 - round_idx / n_total_round)
    return step_size


def calc_ema(org_data, beta, n):
    data = org_data[-n:]
    ema = data[0]
    for d in data[1:]:
        ema = beta * ema + (1 - beta) * d
    return ema
