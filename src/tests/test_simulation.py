import os
import sys
import unittest
import argparse
import logging
import warnings

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from run_simulation import run_simulation

logging.disable(logging.CRITICAL)


src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_project = os.path.dirname(src_path)


class TestSimulator(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter(action="ignore", category=UserWarning)

    def create_args(self):
        test_args = argparse.Namespace()
        test_args.seed = 0
        test_args.gpu_id = None
        test_args.silo_id = None

        test_args.dataset_name = "mnist"
        test_args.model_name = "cnn"
        test_args.n_users = 10
        test_args.n_silos = 3
        test_args.n_silo_per_round = 3
        test_args.user_dist = "uniform-iid"
        test_args.silo_dist = "uniform"
        test_args.user_alpha = 0.5
        test_args.silo_alpha = 0.5
        test_args.n_labels = 10
        test_args.typical_scenario = None

        test_args.n_total_round = 2
        test_args.local_epochs = 2
        test_args.local_learning_rate = 0.05
        test_args.global_learning_rate = 1.0
        test_args.local_batch_size = 64
        test_args.weight_decay = 0.001
        test_args.client_optimizer = "sgd"

        test_args.agg_strategy = "DEFAULT"
        test_args.group_k = 4
        test_args.sigma = 1.0
        test_args.clipping_bound = 1.0
        test_args.delta = 1e-05
        test_args.sampling_rate_q = 0.1
        test_args.validation_ratio = 0.0

        test_args.C_u = None
        test_args.q_u = None
        test_args.epsilon_u = None
        test_args.group_thresholds = None
        test_args.q_step_size = None

        test_args.with_momentum = False
        test_args.off_train_loss_noise = False
        test_args.momentum_weight = 0.9
        test_args.hp_baseline = None
        test_args.step_decay = False
        test_args.initial_q_u = None
        test_args.parallelized = False

        test_args.verbose = 0
        test_args.dry_run = False
        test_args.secure_w = False
        return test_args

    def test_default(self):
        args = self.create_args()
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.9671, 2.3392739817500114),
            "Default simulation has been changed from verified results",
        )

    def test_silo_level(self):
        args = self.create_args()
        args.agg_strategy = "SILO-LEVEL-DP"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.1031, 8469.984008789062),
            "SILO-LEVEL-DP simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][1],
            (1, 7.077391578166641, 1e-05),
            "SILO-LEVEL-DP simulation has been changed from verified results",
        )

    def test_record_level(self):
        args = self.create_args()
        args.agg_strategy = "RECORD-LEVEL-DP"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.7569, 14.955725908279419),
            "RECORD-LEVEL-DP simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][1],
            (1, 1.0164595757401516, 1e-05),
            "RECORD-LEVEL-DP simulation has been changed from verified results",
        )

    def test_uldp_naive(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-NAIVE"
        args.sigma = 0.1
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.134, 116.94588661193848),
            "ULDP-NAIVE simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][1],
            (1, 166.035533599549, 1e-05),
            "ULDP-NAIVE simulation has been changed from verified results",
        )

    def test_uldp_group_k_2(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-GROUP"
        args.group_k = 2
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.1033, 46.13217639923096),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_group_k_8(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-GROUP"
        args.group_k = 8
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.0912, 46.09541726112366),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_group_max(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-GROUP-max"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.7569, 14.955725908279419),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_group_median(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-GROUP-median"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.7474, 15.292479157447815),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_sgd(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-SGD"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.1255, 46.09587359428406),
            "ULDP-SGD simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][1],
            (1, 7.077391578166641, 1e-05),
            "ULDP-SGD simulation has been changed from verified results",
        )

    def test_uldp_sgd_w(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-SGD-w"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.1254, 46.0954704284668),
            "ULDP-SGD simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][1],
            (1, 7.077391578166641, 1e-05),
            "ULDP-SGD simulation has been changed from verified results",
        )

    def test_uldp_avg(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-AVG"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.1734, 41.561235189437866),
            "ULDP-AVG simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][1],
            (1, 7.077391578166641, 1e-05),
            "ULDP-AVG simulation has been changed from verified results",
        )

    def test_uldp_avg_w(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-AVG-w"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.1756, 41.55921483039856),
            "ULDP-AVG-w simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][1],
            (1, 7.077391578166641, 1e-05),
            "ULDP-AVG-w simulation has been changed from verified results",
        )

    def test_uldp_avg_ws(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-AVG-ws"
        args.sampling_rate_q = 0.1
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.1086, 20597.387268066406),
            "ULDP-AVG-ws simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][1],
            (1, 2.412889750604903, 1e-05),
            "ULDP-AVG-ws simulation has been changed from verified results",
        )


if __name__ == "__main__":
    unittest.main()
