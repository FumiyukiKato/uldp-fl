import unittest


from coordinator import Coordinator

seed = 0


class TestCoodinator(unittest.TestCase):
    def test_build_user_bound_histograms_random(self):
        n_users = 10
        n_silos = 3
        group_k = 5
        cood = Coordinator(base_seed=seed, n_silos=n_silos, n_users=n_users)
        user_bound_histograms = cood.build_user_bound_histograms(
            group_k=group_k,
        )
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
        cood = Coordinator(base_seed=seed, n_silos=n_silos, n_users=n_users)
        user_histogram_dct = {
            0: {0: 1, 1: 3, 2: 5, 4: 1},
            1: {0: 1, 1: 3, 2: 5, 4: 1},
            2: {0: 3, 1: 2, 3: 3, 4: 1, 5: 5},
        }
        user_bound_histograms = cood.build_user_bound_histograms(
            group_k=group_k,
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
        cood = Coordinator(base_seed=seed, n_silos=n_silos, n_users=n_users)
        user_histogram_dct = {
            0: {0: 1, 1: 3, 2: 5, 4: 1},
            1: {0: 1, 1: 3, 2: 5, 4: 1},
            2: {0: 3, 1: 2, 3: 3, 4: 1, 5: 5},
        }
        user_bound_histograms = cood.build_user_bound_histograms(
            group_k=group_k,
            old_user_histogram_dct=user_histogram_dct,
            minimum_number_of_records=min_value,
        )
        self.assertDictEqual(
            user_bound_histograms,
            {0: {0: 2, 1: 2, 2: 2}, 1: {0: 2, 2: 2, 4: 2}, 2: {1: 2, 4: 2, 3: 4, 5: 4}},
        )


if __name__ == "__main__":
    unittest.main()
