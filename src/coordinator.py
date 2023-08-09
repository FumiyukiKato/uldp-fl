from typing import Dict, Optional
import numpy as np

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
    ):
        self.random_state = np.random.RandomState(seed=base_seed + 2000000)
        self.n_users = n_users
        self.n_silos = n_silos
        self.group_k = group_k
        self.sampling_rate_q = sampling_rate_q
        self.agg_strategy = agg_strategy
        self.ready_silos = set()
        self.original_user_hist_dct = {silo_id: {} for silo_id in range(n_silos)}

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
        self, weighted: bool = False, is_sample: bool = False
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
            sampled_user_ids = user_ids[
                self.random_state.rand(len(user_ids)) < self.sampling_rate_q
            ]
            sampled_user_ids_set = set(sampled_user_ids)
            for silo_id in range(self.n_silos):
                for user_id in range(self.n_users):
                    if user_id not in sampled_user_ids_set:
                        user_weights_per_silo[silo_id][user_id] = 0.0

        return user_weights_per_silo

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
