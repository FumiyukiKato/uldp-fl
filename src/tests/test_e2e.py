import os
import unittest
import argparse

import warnings
from run_simulation import run_simulation

import logging

logging.disable(logging.CRITICAL)


class TestSimulater(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter(action="ignore", category=UserWarning)
        print("\n\n============== Simulation Running ... ===============")

    def create_args(self):
        test_args = argparse.Namespace()
        test_args.seed = 0
        test_args.gpu_id = None
        test_args.silo_id = None

        test_args.dataset_name = "mnist"
        test_args.model_name = "cnn"
        test_args.n_users = 10000
        test_args.n_silos = 3
        test_args.user_dist = "uniform-iid"
        test_args.silo_dist = "uniform"
        test_args.user_alpha = 0.5
        test_args.silo_alpha = 0.5
        test_args.n_labels = 10
        test_args.typical_scenaio = None

        test_args.n_silo_per_round = 2
        test_args.n_total_round = 3
        test_args.epochs = 5
        test_args.learning_rate = 0.05
        test_args.local_batch_size = 64
        test_args.weight_decay = 0.001
        test_args.client_optimizer = "sgd"

        test_args.agg_strategy = "DEFAULT"
        test_args.group_k = 4
        test_args.sigma = 1.0
        test_args.clipping_bound = 1.0
        test_args.delta = 1e-05
        test_args.sampling_rate_q = None

        test_args.verbose = 0
        return test_args

    def test_default(self):
        args = self.create_args()
        path_project = os.path.abspath("..")
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][2],
            (2, 0.9805, 63.813511928442495),
            "Default simulation has been changed from verified results",
        )

    def test_silo_level(self):
        args = self.create_args()
        path_project = os.path.abspath("..")
        args.agg_strategy = "SILO-LEVEL-DP"
        args.clipping_bound = 1.0
        args.sigma = 1.0
        args.local_batch_size = 32
        args.epochs = 10
        args.n_silos = 1000
        args.n_silo_per_round = 100
        args.n_total_round = 5
        args.learning_rate = 0.1
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.8689, 552.7958926409483),
            "SILO-LEVEL-DP simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 2.9021155032398083, 1e-05),
            "SILO-LEVEL-DP simulation has been changed from verified results",
        )

    def test_record_level(self):
        args = self.create_args()
        path_project = os.path.abspath("..")
        args.agg_strategy = "RECORD-LEVEL-DP"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][2],
            (2, 0.8733, 498.06457424448126),
            "RECORD-LEVEL-DP simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][2],
            (2, 1.4223797945630121, 1e-05),
            "RECORD-LEVEL-DP simulation has been changed from verified results",
        )

    def test_uldp_naive(self):
        args = self.create_args()
        path_project = os.path.abspath("..")
        args.agg_strategy = "ULDP-NAIVE"
        args.clipping_bound = 1.0
        args.sigma = 0.4
        args.local_batch_size = 32
        args.epochs = 10
        args.n_silos = 1000
        args.n_silo_per_round = 100
        args.n_total_round = 5
        args.learning_rate = 0.1
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.6742, 1063.3421179056168),
            "ULDP-NAIVE simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 40.970493283868805, 1e-05),
            "ULDP-NAIVE simulation has been changed from verified results",
        )

    def test_uldp_group_k_2(self):
        args = self.create_args()
        path_project = os.path.abspath("..")
        args.agg_strategy = "ULDP-GROUP"
        args.group_k = 2
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][2],
            (2, 0.8009, 625.2333497703075),
            "ULDP-GROUP simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][2],
            (2, 6.696719892919694, 1e-05),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_group_k_4(self):
        args = self.create_args()
        path_project = os.path.abspath("..")
        args.agg_strategy = "ULDP-GROUP"
        args.group_k = 4
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][2],
            (2, 0.8588, 501.83081338735064),
            "ULDP-GROUP simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][2],
            (2, 13.330380328831556, 1e-05),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_group_k_8(self):
        args = self.create_args()
        path_project = os.path.abspath("..")
        args.agg_strategy = "ULDP-GROUP"
        args.group_k = 8
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][2],
            (2, 0.8733, 498.06457424448126),
            "ULDP-GROUP simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][2],
            (2, 232374.77329073567, 1e-05),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_sgd(self):
        args = self.create_args()
        path_project = os.path.abspath("..")
        args.agg_strategy = "ULDP-SGD"
        args.learning_rate = 5.0
        args.sigma = 8.0
        args.n_total_round = 3
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][2],
            (2, 0.2793, 2238.972419977188),
            "ULDP-SGD simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][2],
            (2, 0.865730031476462, 1e-05),
            "ULDP-SGD simulation has been changed from verified results",
        )

    def test_uldp_avg(self):
        args = self.create_args()
        path_project = os.path.abspath("..")
        args.agg_strategy = "ULDP-AVG"
        args.n_users = 10000
        args.clipping_bound = 10.0
        args.learning_rate = 0.1
        args.epochs = 1
        args.sigma = 0.01
        args.n_total_round = 20
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][2],
            (2, 0.0928, 2306.6163182258606),
            "ULDP-SGD simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][2],
            (2, 16611.77825757886, 1e-05),
            "ULDP-SGD simulation has been changed from verified results",
        )


if __name__ == "__main__":
    unittest.main()
