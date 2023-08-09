import unittest
import os
import sys

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from coordinator import Coordinator

seed = 0


class TestCoodinator(unittest.TestCase):
    def test_build_user_weights(self):
        n_users = 10
        n_silos = 3
        cood = Coordinator(base_seed=seed, n_silos=n_silos, n_users=n_users)
        user_hist_per_silo = {
            0: {0: 3, 1: 1, 3: 3, 4: 1, 6: 2, 7: 2, 8: 3, 9: 2},
            1: {1: 2, 2: 5, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2},
            2: {0: 2, 1: 2, 4: 2, 5: 3, 9: 3},
        }
        for silo_id in user_hist_per_silo:
            cood.set_user_hist_by_silo_id(silo_id, user_hist_per_silo[silo_id])
        user_weights = cood.build_user_weights(weighted=True)
        self.assertDictEqual(
            user_weights,
            {
                0: {
                    0: 0.6,
                    1: 0.2,
                    2: 0.0,
                    3: 0.6,
                    4: 0.2,
                    5: 0.0,
                    6: 0.4,
                    7: 0.4,
                    8: 0.6,
                    9: 0.4,
                },
                1: {
                    0: 0.0,
                    1: 0.4,
                    2: 1.0,
                    3: 0.4,
                    4: 0.4,
                    5: 0.4,
                    6: 0.6,
                    7: 0.6,
                    8: 0.4,
                    9: 0.0,
                },
                2: {
                    0: 0.4,
                    1: 0.4,
                    2: 0.0,
                    3: 0.0,
                    4: 0.4,
                    5: 0.6,
                    6: 0.0,
                    7: 0.0,
                    8: 0.0,
                    9: 0.6,
                },
            },
        )

    def test_build_user_weights_with_user_level_sampling(self):
        n_users = 10
        n_silos = 3
        seed = 100
        sampling_rate_q = 0.3
        cood = Coordinator(
            base_seed=seed,
            n_silos=n_silos,
            n_users=n_users,
            sampling_rate_q=sampling_rate_q,
        )
        user_hist_per_silo = {
            0: {0: 3, 1: 1, 3: 3, 4: 1, 6: 2, 7: 2, 8: 3, 9: 2},
            1: {1: 2, 2: 5, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2},
            2: {0: 2, 1: 2, 4: 2, 5: 3, 9: 3},
        }
        for silo_id in user_hist_per_silo:
            cood.set_user_hist_by_silo_id(silo_id, user_hist_per_silo[silo_id])
        user_weights = cood.build_user_weights(weighted=True, is_sample=True)
        self.assertDictEqual(
            user_weights,
            {
                0: {
                    0: 0.0,
                    1: 0.2,
                    2: 0.0,
                    3: 0.0,
                    4: 0.0,
                    5: 0.0,
                    6: 0.0,
                    7: 0.0,
                    8: 0.0,
                    9: 0.4,
                },
                1: {
                    0: 0.0,
                    1: 0.4,
                    2: 0.0,
                    3: 0.0,
                    4: 0.0,
                    5: 0.4,
                    6: 0.0,
                    7: 0.0,
                    8: 0.0,
                    9: 0.0,
                },
                2: {
                    0: 0.0,
                    1: 0.4,
                    2: 0.0,
                    3: 0.0,
                    4: 0.0,
                    5: 0.6,
                    6: 0.0,
                    7: 0.0,
                    8: 0.0,
                    9: 0.6,
                },
            },
        )

    def test_build_user_weights_random(self):
        n_users = 10
        n_silos = 3
        cood = Coordinator(base_seed=seed, n_silos=n_silos, n_users=n_users)
        user_hist_per_silo = {
            0: {0: 3, 1: 1, 3: 3, 4: 1, 6: 2, 7: 2, 8: 3, 9: 2},
            1: {1: 2, 2: 5, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2},
            2: {0: 2, 1: 2, 4: 2, 5: 3, 9: 3},
        }
        for silo_id in user_hist_per_silo:
            cood.set_user_hist_by_silo_id(silo_id, user_hist_per_silo[silo_id])
        user_weights = cood.build_user_weights(weighted=False)
        self.assertDictEqual(
            user_weights,
            {
                0: {
                    0: 0.3333333333333333,
                    1: 0.3333333333333333,
                    2: 0.3333333333333333,
                    3: 0.3333333333333333,
                    4: 0.3333333333333333,
                    5: 0.3333333333333333,
                    6: 0.3333333333333333,
                    7: 0.3333333333333333,
                    8: 0.3333333333333333,
                    9: 0.3333333333333333,
                },
                1: {
                    0: 0.3333333333333333,
                    1: 0.3333333333333333,
                    2: 0.3333333333333333,
                    3: 0.3333333333333333,
                    4: 0.3333333333333333,
                    5: 0.3333333333333333,
                    6: 0.3333333333333333,
                    7: 0.3333333333333333,
                    8: 0.3333333333333333,
                    9: 0.3333333333333333,
                },
                2: {
                    0: 0.3333333333333333,
                    1: 0.3333333333333333,
                    2: 0.3333333333333333,
                    3: 0.3333333333333333,
                    4: 0.3333333333333333,
                    5: 0.3333333333333333,
                    6: 0.3333333333333333,
                    7: 0.3333333333333333,
                    8: 0.3333333333333333,
                    9: 0.3333333333333333,
                },
            },
        )

    def test_build_user_weights_random_with_sampling(self):
        n_users = 10
        n_silos = 3
        seed = 100
        sampling_rate_q = 0.3
        cood = Coordinator(
            base_seed=seed,
            n_silos=n_silos,
            n_users=n_users,
            sampling_rate_q=sampling_rate_q,
        )
        user_hist_per_silo = {
            0: {0: 3, 1: 1, 3: 3, 4: 1, 6: 2, 7: 2, 8: 3, 9: 2},
            1: {1: 2, 2: 5, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2},
            2: {0: 2, 1: 2, 4: 2, 5: 3, 9: 3},
        }
        for silo_id in user_hist_per_silo:
            cood.set_user_hist_by_silo_id(silo_id, user_hist_per_silo[silo_id])
        user_weights = cood.build_user_weights(weighted=False, is_sample=True)
        self.assertDictEqual(
            user_weights,
            {
                0: {
                    0: 0.0,
                    1: 0.3333333333333333,
                    2: 0.0,
                    3: 0.0,
                    4: 0.0,
                    5: 0.3333333333333333,
                    6: 0.0,
                    7: 0.0,
                    8: 0.0,
                    9: 0.3333333333333333,
                },
                1: {
                    0: 0.0,
                    1: 0.3333333333333333,
                    2: 0.0,
                    3: 0.0,
                    4: 0.0,
                    5: 0.3333333333333333,
                    6: 0.0,
                    7: 0.0,
                    8: 0.0,
                    9: 0.3333333333333333,
                },
                2: {
                    0: 0.0,
                    1: 0.3333333333333333,
                    2: 0.0,
                    3: 0.0,
                    4: 0.0,
                    5: 0.3333333333333333,
                    6: 0.0,
                    7: 0.0,
                    8: 0.0,
                    9: 0.3333333333333333,
                },
            },
        )

    def test_build_user_bound_histograms_random(self):
        n_users = 10
        n_silos = 3
        group_k = 5
        cood = Coordinator(
            base_seed=seed, n_silos=n_silos, n_users=n_users, group_k=group_k
        )
        user_bound_histograms = cood.build_user_bound_histograms()
        self.assertDictEqual(
            user_bound_histograms,
            {
                0: {0: 3, 1: 1, 3: 3, 4: 1, 6: 2, 7: 2, 8: 3, 9: 2},
                1: {1: 2, 2: 5, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2},
                2: {0: 2, 1: 2, 4: 2, 5: 3, 9: 3},
            },
        )

    def test_build_user_bound_histograms_given_user_histogram_dct(self):
        n_users = 10
        n_silos = 3
        group_k = 3
        cood = Coordinator(
            base_seed=seed, n_silos=n_silos, n_users=n_users, group_k=group_k
        )
        user_histogram_dct = {
            0: {0: 1, 1: 3, 2: 5, 4: 1},
            1: {0: 1, 1: 3, 2: 5, 4: 1},
            2: {0: 3, 1: 2, 3: 3, 4: 1, 5: 5},
        }
        user_bound_histograms = cood.build_user_bound_histograms(
            old_user_histogram_dct=user_histogram_dct,
        )
        self.assertDictEqual(
            user_bound_histograms,
            {
                0: {0: 1, 1: 1, 2: 2, 4: 1},
                1: {0: 1, 1: 1, 2: 1, 4: 1},
                2: {0: 1, 1: 1, 3: 3, 4: 1, 5: 3},
            },
        )

    def test_build_user_bound_histograms_given_user_histogram_dct_given_min_value(self):
        n_users = 10
        n_silos = 3
        group_k = 4
        min_value = 2
        cood = Coordinator(
            base_seed=seed, n_silos=n_silos, n_users=n_users, group_k=group_k
        )
        user_histogram_dct = {
            0: {0: 1, 1: 3, 2: 5, 4: 1},
            1: {0: 1, 1: 3, 2: 5, 4: 1},
            2: {0: 3, 1: 2, 3: 3, 4: 1, 5: 5},
        }
        user_bound_histograms = cood.build_user_bound_histograms(
            old_user_histogram_dct=user_histogram_dct,
            minimum_number_of_records=min_value,
        )
        self.assertDictEqual(
            user_bound_histograms,
            {0: {0: 2, 1: 2, 2: 2}, 1: {0: 2, 2: 2, 4: 2}, 2: {1: 2, 4: 2, 3: 4, 5: 4}},
        )


if __name__ == "__main__":
    unittest.main()
