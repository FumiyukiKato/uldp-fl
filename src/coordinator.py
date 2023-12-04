from typing import Dict, Optional, List, Tuple
import numpy as np
import copy
from opacus.accountants import RDPAccountant

from mylogger import logger


class Coordinator:
    """
    Coordinator class for federated learning.
    Receive user distribution from silos and decide weighting strategies on shared paramters for each silo.
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

        if self.agg_strategy == "PULDP-AVG-online":
            assert (
                q_step_size is not None and delta is not None and sigma is not None
            ), "q_step_size, delta, sigma must be given for PULDP-AVG-online"
            self.q_step_size = q_step_size
            self.delta, self.sigma, self.n_total_round = delta, sigma, n_total_round
            self.epsilon_groups = group_by_closest_below(
                epsilon_u_dct=epsilon_u, group_thresholds=group_thresholds
            )
            self.hp_dct_by_eps: Dict[float, Tuple] = {}
            self.param_history: Dict[float, List] = {}
            self.loss_history: Dict[float, List] = {}

            INITIAL_Q_U = 1.0
            for eps_u in self.epsilon_groups.keys():
                initial_C_u, eps = from_q_u(
                    q_u=INITIAL_Q_U,
                    delta=self.delta,
                    epsilon_u=eps_u,
                    sigma=self.sigma,
                    T=self.n_total_round,
                )
                self.hp_dct_by_eps[eps_u] = (INITIAL_Q_U, initial_C_u)
                self.param_history[eps_u] = [(INITIAL_Q_U, initial_C_u)]
                self.loss_history[eps_u] = []

    def set_user_hist_by_silo_id(self, silo_id: int, user_hist: Dict):
        self.ready_silos.add(silo_id)
        self.original_user_hist_dct[silo_id] = user_hist

    def is_ready(self) -> bool:
        if len(self.ready_silos) == self.n_silos:
            if self.agg_strategy == "ULDP-GROUP-max":
                group_max = self.get_group_max()
                logger.info(f"Group max: {group_max}")
                self.set_group_k(group_max)
            elif self.agg_strategy == "ULDP-GROUP-median":
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
            if self.agg_strategy == "PULDP-AVG":
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
    ) -> Tuple[
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

        stepped_user_weights_per_silo = copy.deepcopy(user_weights_per_silo)

        user_ids = np.array(range(self.n_users))

        self.q_u_list = np.zeros(self.n_users)
        self.C_u_list = np.zeros(self.n_users)
        self.stepped_q_u_list = np.zeros(self.n_users)
        self.stepped_C_u_list = np.zeros(self.n_users)
        for eps_u, user_ids_per_eps in self.epsilon_groups.items():
            q_u, C_u = self.hp_dct_by_eps[eps_u]
            stepped_q_u, stepped_C_u = compute_stepped_qC(
                step_size=self.q_step_size,
                q_u=q_u,
                delta=self.delta,
                eps_u=eps_u,
                sigma=self.sigma,
                T=self.n_total_round,
            )
            self.q_u_list[user_ids_per_eps] = q_u
            self.C_u_list[user_ids_per_eps] = C_u
            self.stepped_q_u_list[user_ids_per_eps] = stepped_q_u
            self.stepped_C_u_list[user_ids_per_eps] = stepped_C_u

        sampled_user_ids = user_ids[
            self.random_state.rand(len(self.q_u_list)) < self.q_u_list
        ]
        stepped_sampled_user_ids = user_ids[
            self.random_state.rand(len(self.stepped_q_u_list)) < self.stepped_q_u_list
        ]

        sampled_user_ids_set = set(sampled_user_ids)
        stepped_sampled_user_ids_set = set(stepped_sampled_user_ids)
        for silo_id in range(self.n_silos):
            for user_id in range(self.n_users):
                if user_id not in sampled_user_ids_set:
                    user_weights_per_silo[silo_id][user_id] = 0.0
                if user_id not in stepped_sampled_user_ids_set:
                    stepped_user_weights_per_silo[silo_id][user_id] = 0.0

        return (
            user_weights_per_silo,
            self.C_u_list,
            stepped_user_weights_per_silo,
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

    def online_optimize(self, loss_diff_dct: Dict[float, float]):
        for eps_u in self.epsilon_groups.keys():
            loss_diff = loss_diff_dct[eps_u]  # diff = stepped_test_loss - test_loss
            q_u, _ = self.hp_dct_by_eps[eps_u]
            if loss_diff < 0:
                q_u = q_u * self.q_step_size
            else:
                q_u = q_u / self.q_step_size
                q_u = min(q_u, 1.0)
            C_u, eps = from_q_u(q_u, self.delta, eps_u, self.sigma, self.n_total_round)
            self.hp_dct_by_eps[eps_u] = (q_u, C_u)
            self.param_history[eps_u].append((q_u, C_u))
            self.loss_history[eps_u].append(loss_diff)


def group_by_closest_below(epsilon_u_dct: Dict, group_thresholds: List):
    minimum = min(epsilon_u_dct.values())
    group_thresholds = set(group_thresholds) | {minimum}
    grouped = {
        g: [] for g in group_thresholds
    }  # Initialize the dictionary with empty lists for each group threshold
    for key, value in epsilon_u_dct.items():
        # Find the closest group threshold that is less than or equal to the value
        closest_group = max([g for g in group_thresholds if g <= value], default=None)
        # If a suitable group is found, append the key to the corresponding list
        if closest_group is not None:
            grouped[closest_group].append(key)

    return grouped


# binary search given q_u
def from_q_u(q_u, delta, epsilon_u, sigma, T, m=100, precision=1e-6):
    max_sensitivity_u = 100
    min_sensitivity_u = 0
    while True:
        sensitivity_u = (max_sensitivity_u + min_sensitivity_u) / 2
        # func_gaussian = lambda x: RDP_gaussian_with_C(sigma, x, sensitivity_u)
        # accountant = rdp_acct.anaRDPacct(m=m)
        accountant = RDPAccountant()
        for i in range(T):
            accountant.step(noise_multiplier=sigma / sensitivity_u, sample_rate=q_u)
            # accountant.compose_subsampled_mechanisms_lowerbound(func=func_gaussian, prob=q_u)
        # eps = accountant.get_eps(delta)
        eps = accountant.get_epsilon(delta=delta)
        if eps < epsilon_u:
            min_sensitivity_u = sensitivity_u
        else:
            max_sensitivity_u = sensitivity_u
        if 0 < epsilon_u - eps and epsilon_u - eps < precision:
            return sensitivity_u, eps


def compute_stepped_qC(
    step_size: float, q_u: float, delta: float, eps_u: float, sigma: float, T: int
):
    dst_q = q_u * (step_size)
    dst_C, eps = from_q_u(q_u=dst_q, delta=delta, epsilon_u=eps_u, sigma=sigma, T=T)
    return dst_q, dst_C
