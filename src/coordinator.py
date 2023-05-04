from typing import Dict
import numpy as np


class Coordinator:
    """
    Coordinator class for federated learning.
    Receive user distribution from silos and decide strategies.
    """

    def __init__(
        self,
        base_seed: int,
        n_silos: int,
        n_users: int,
    ):
        self.random_state = np.random.RandomState(seed=base_seed + 2000000)
        self.n_users = n_users
        self.n_silos = n_silos
        self.original_user_hist_dct = {silo_id: {} for silo_id in range(n_silos)}

    def build_user_weights(self, uniform: bool = True) -> Dict[int, Dict[int, float]]:
        """
        Build user weights for ULDP-SGD.
        """
        user_weights_per_silo = {silo_id: {} for silo_id in range(self.n_silos)}
        if uniform:
            for silo_id in range(self.n_silos):
                user_weights_per_silo[silo_id] = {
                    user_id: 1.0 / self.n_silos for user_id in range(self.n_users)
                }
        else:
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
        return user_weights_per_silo

    def build_user_bound_histograms(
        self,
        group_k: int,
        old_user_histogram_dct: Dict[int, Dict[int, int]] = None,
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
                while current_user_count < group_k and current_user_count < user_count:
                    if (
                        user_id in old_user_histogram_dct[round_robin_silo_idx]
                        and old_user_histogram_dct[round_robin_silo_idx][user_id] > 0
                    ):
                        old_user_histogram_dct[round_robin_silo_idx][user_id] -= 1
                        if user_id not in new_user_histogram_dct[round_robin_silo_idx]:
                            new_user_histogram_dct[round_robin_silo_idx][user_id] = 1
                        else:
                            new_user_histogram_dct[round_robin_silo_idx][user_id] += 1
                        current_user_count += 1
                    round_robin_silo_idx = (round_robin_silo_idx + 1) % len(
                        old_user_histogram_dct
                    )
            return new_user_histogram_dct

        else:
            # Randomly assign group_k capacity for users to each silo
            new_user_histogram_dct = {silo_id: {} for silo_id in range(self.n_silos)}
            count_matrix = self.random_state.choice(
                self.n_silos, (self.n_users, group_k), replace=True
            )
            for user_id in range(self.n_users):
                for silo_id in count_matrix[user_id]:
                    if user_id not in new_user_histogram_dct[silo_id]:
                        new_user_histogram_dct[silo_id][user_id] = 1
                    else:
                        new_user_histogram_dct[silo_id][user_id] += 1
            return new_user_histogram_dct
