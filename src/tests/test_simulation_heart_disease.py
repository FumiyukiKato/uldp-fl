import os
import sys
import unittest
import argparse
import logging
import warnings

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from run_simulation import run_simulation
from mylogger import logger_set_debug

logging.disable(logging.CRITICAL)
# logger_set_debug()

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

        test_args.dataset_name = "heart_disease"
        test_args.model_name = "cnn"
        test_args.n_users = 300
        test_args.n_silos = 4
        test_args.n_silo_per_round = 4
        test_args.user_dist = "uniform-iid"
        test_args.silo_dist = "uniform"
        test_args.user_alpha = 0.5
        test_args.silo_alpha = 0.5
        test_args.n_labels = 2

        test_args.n_total_round = 5
        test_args.local_epochs = 20
        test_args.local_learning_rate = 0.001
        test_args.global_learning_rate = 10.0
        test_args.local_batch_size = 4
        test_args.weight_decay = 0.001
        test_args.client_optimizer = "sgd"

        test_args.agg_strategy = "DEFAULT"
        test_args.group_k = 4
        test_args.sigma = 1.0
        test_args.clipping_bound = 0.1
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
        test_args.dynamic_global_learning_rate = False

        test_args.verbose = 0
        test_args.dry_run = False
        test_args.secure_w = False
        return test_args

    def test_default(self):
        args = self.create_args()
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.7834645669291339, 0.4788423776626587),
            "Default simulation has been changed from verified results",
        )

    def test_silo_level(self):
        args = self.create_args()
        args.agg_strategy = "SILO-LEVEL-DP"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.7637795275590551, 0.5708376169204712),
            "SILO-LEVEL-DP simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 12.301691480042896, 1e-05),
            "SILO-LEVEL-DP simulation has been changed from verified results",
        )

    def test_record_level(self):
        args = self.create_args()
        args.agg_strategy = "RECORD-LEVEL-DP"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.8110236220472441, 0.4464093744754791),
            "RECORD-LEVEL-DP simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 30.83078518806741, 1e-05),
            "RECORD-LEVEL-DP simulation has been changed from verified results",
        )

    def test_uldp_naive(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-NAIVE"
        args.sigma = 0.1
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.7716535433070866, 0.49562427401542664),
            "ULDP-NAIVE simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 354.86126007165325, 1e-05),
            "ULDP-NAIVE simulation has been changed from verified results",
        )

    def test_uldp_group_k_2(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-GROUP"
        args.group_k = 2
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.8070866141732284, 0.4752809405326843),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_group_k_8(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-GROUP"
        args.group_k = 8
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.8110236220472441, 0.4464093744754791),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_group_max(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-GROUP-max"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.8110236220472441, 0.4464093744754791),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_group_median(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-GROUP-median"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.8070866141732284, 0.4752809405326843),
            "ULDP-GROUP simulation has been changed from verified results",
        )

    def test_uldp_sgd(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-SGD"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.6732283464566929, 0.6535313129425049),
            "ULDP-SGD simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 12.301691480042896, 1e-05),
            "ULDP-SGD simulation has been changed from verified results",
        )

    def test_uldp_sgd_w(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-SGD-w"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.7401574803149606, 0.581291675567627),
            "ULDP-SGD simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 12.301691480042896, 1e-05),
            "ULDP-SGD simulation has been changed from verified results",
        )

    def test_uldp_avg(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-AVG"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.5866141732283464, 0.6879163384437561),
            "ULDP-AVG simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 12.301691480042896, 1e-05),
            "ULDP-AVG simulation has been changed from verified results",
        )

    def test_uldp_avg_w(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-AVG-w"
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.6535433070866141, 0.660047709941864),
            "ULDP-AVG-w simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 12.301691480042896, 1e-05),
            "ULDP-AVG-w simulation has been changed from verified results",
        )

    def test_uldp_avg_ws(self):
        args = self.create_args()
        args.agg_strategy = "ULDP-AVG-ws"
        args.sampling_rate_q = 0.1
        results = run_simulation(args, path_project)
        self.assertTupleEqual(
            results["global"]["global_test"][4],
            (4, 0.6614173228346457, 0.6584948897361755),
            "ULDP-AVG-ws simulation has been changed from verified results",
        )
        self.assertTupleEqual(
            results["global"]["privacy_budget"][4],
            (4, 2.9021155032398083, 1e-05),
            "ULDP-AVG-ws simulation has been changed from verified results",
        )

    def test_puldp_avg(self):
        args = self.create_args()
        args.agg_strategy = "PULDP-AVG"
        args.epsilon_list = [0.15, 3.0, 5.0]
        args.group_thresholds = [0.15, 3.0, 5.0]
        args.ratio_list = [0.6, 0.25, 0.15]
        args.validation_ratio = 0.5
        args.n_total_round = 2
        static_q_u_list = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
        best_idx_per_group = {0.15: 6, 3.0: 3, 5.0: 0}
        results = run_simulation(
            args,
            path_project,
            epsilon_list=args.epsilon_list,
            ratio_list=args.ratio_list,
            q_step_size=args.q_step_size,
            idx_per_group=best_idx_per_group,
            static_q_u_list=static_q_u_list,
        )
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.5433070866141733, 0.6876358985900879),
            "PUDLP-AVG simulation has been changed from verified results",
        )

    def test_puldp_qctest(self):
        args = self.create_args()
        args.agg_strategy = "PULDP-AVG-QCTest"
        args.epsilon_list = [0.15, 3.0, 5.0]
        args.group_thresholds = [0.15, 3.0, 5.0]
        args.ratio_list = [0.6, 0.25, 0.15]
        args.validation_ratio = 0.5
        args.n_total_round = 2
        args.initial_q_u = 0.5
        args.q_step_size = 0.8
        results = run_simulation(
            args,
            path_project,
            epsilon_list=args.epsilon_list,
            ratio_list=args.ratio_list,
        )
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.5433070866141733, 0.6915268898010254),
            "QC Test simulation has been changed from verified results",
        )
        self.assertDictEqual(
            results["global"]["final_eps"],
            {0.15: 0.14999975234186405, 3.0: 2.999999771958145, 5.0: 4.999999584618626},
            "PUDLP-AVG simulation has been changed from verified results",
        )

    def test_puldp_qctrain(self):
        args = self.create_args()
        args.agg_strategy = "PULDP-AVG-QCTrain"
        args.epsilon_list = [0.15, 3.0, 5.0]
        args.group_thresholds = [0.15, 3.0, 5.0]
        args.ratio_list = [0.6, 0.25, 0.15]
        args.validation_ratio = 0.5
        args.n_total_round = 2
        args.initial_q_u = 0.5
        args.q_step_size = 0.8
        results = run_simulation(
            args,
            path_project,
            epsilon_list=args.epsilon_list,
            ratio_list=args.ratio_list,
        )
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.5748031496062992, 0.6855200529098511),
            "QC Train simulation has been changed from verified results",
        )
        self.assertDictEqual(
            results["global"]["final_eps"],
            {0.15: 0.1499995968601459, 3.0: 2.999999431513258, 5.0: 4.999999168837267},
            "QC Test simulation has been changed from verified results",
        )

    def test_puldp_qctrain_dynamic(self):
        args = self.create_args()
        args.agg_strategy = "PULDP-AVG-QCTrain"
        args.epsilon_list = [0.15, 3.0, 5.0]
        args.group_thresholds = [0.15, 3.0, 5.0]
        args.ratio_list = [0.6, 0.25, 0.15]
        args.validation_ratio = 0.5
        args.n_total_round = 2
        args.initial_q_u = 0.5
        args.q_step_size = 0.8
        args.dynamic_global_learning_rate = True
        results = run_simulation(
            args,
            path_project,
            epsilon_list=args.epsilon_list,
            ratio_list=args.ratio_list,
        )
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.6771653543307087, 0.6292539834976196),
            "QC Train (dynamic global LR) simulation has been changed from verified results",
        )
        self.assertDictEqual(
            results["global"]["final_eps"],
            {0.15: 0.1499995968601459, 3.0: 2.999999431513258, 5.0: 4.999999168837267},
            "QC Train (dynamic global LR) simulation has been changed from verified results",
        )

    def test_puldp_qctrain_random_updown(self):
        args = self.create_args()
        args.agg_strategy = "PULDP-AVG-QCTrain"
        args.epsilon_list = [0.15, 3.0, 5.0]
        args.group_thresholds = [0.15, 3.0, 5.0]
        args.ratio_list = [0.6, 0.25, 0.15]
        args.validation_ratio = 0.5
        args.n_total_round = 2
        args.initial_q_u = 0.5
        args.q_step_size = 0.8
        args.hp_baseline = "random-updown"
        results = run_simulation(
            args,
            path_project,
            epsilon_list=args.epsilon_list,
            ratio_list=args.ratio_list,
        )
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.5511811023622047, 0.6850407123565674),
            "QC Train (baseline random-updown) simulation has been changed from verified results",
        )
        self.assertDictEqual(
            results["global"]["final_eps"],
            {0.15: 0.149999489503169, 3.0: 2.9999996164863876, 5.0: 4.999999353778011},
            "QC Train (baseline random-updown) simulation has been changed from verified results",
        )

    def test_puldp_qctrain_random_log(self):
        args = self.create_args()
        args.agg_strategy = "PULDP-AVG-QCTrain"
        args.epsilon_list = [0.15, 3.0, 5.0]
        args.group_thresholds = [0.15, 3.0, 5.0]
        args.ratio_list = [0.6, 0.25, 0.15]
        args.validation_ratio = 0.5
        args.n_total_round = 2
        args.initial_q_u = 0.5
        args.q_step_size = 0.8
        args.hp_baseline = "random-log"
        results = run_simulation(
            args,
            path_project,
            epsilon_list=args.epsilon_list,
            ratio_list=args.ratio_list,
        )
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.5905511811023622, 0.6790423393249512),
            "QC Train (baseline random-log) simulation has been changed from verified results",
        )
        self.assertDictEqual(
            results["global"]["final_eps"],
            {0.15: 0.14999932099027585, 3.0: 2.999999551003775, 5.0: 4.999999705467997},
            "QC Train (baseline random-log) simulation has been changed from verified results",
        )

    def test_puldp_qctrain_random(self):
        args = self.create_args()
        args.agg_strategy = "PULDP-AVG-QCTrain"
        args.epsilon_list = [0.15, 3.0, 5.0]
        args.group_thresholds = [0.15, 3.0, 5.0]
        args.ratio_list = [0.6, 0.25, 0.15]
        args.validation_ratio = 0.5
        args.n_total_round = 2
        args.initial_q_u = 0.5
        args.q_step_size = 0.8
        args.hp_baseline = "random"
        results = run_simulation(
            args,
            path_project,
            epsilon_list=args.epsilon_list,
            ratio_list=args.ratio_list,
        )
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.5511811023622047, 0.6837267875671387),
            "QC Train (baseline random) simulation has been changed from verified results",
        )
        self.assertDictEqual(
            results["global"]["final_eps"],
            {0.15: 0.1499994179954894, 3.0: 2.999999783932882, 5.0: 4.999999232325652},
            "QC Train (baseline random) simulation has been changed from verified results",
        )

    def test_puldp_qctrain_random_parallel(self):
        args = self.create_args()
        args.agg_strategy = "PULDP-AVG-QCTrain"
        args.epsilon_list = [0.15, 3.0, 5.0]
        args.group_thresholds = [0.15, 3.0, 5.0]
        args.ratio_list = [0.6, 0.25, 0.15]
        args.validation_ratio = 0.5
        args.n_total_round = 2
        args.initial_q_u = 0.5
        args.q_step_size = 0.8
        args.hp_baseline = "random"
        args.parallelized = True
        results = run_simulation(
            args,
            path_project,
            epsilon_list=args.epsilon_list,
            ratio_list=args.ratio_list,
        )
        self.assertTupleEqual(
            results["global"]["global_test"][1],
            (1, 0.5511811023622047, 0.68278568983078),
            "Parallelized QC Train (baseline random) simulation has been changed from verified results",
        )
        self.assertDictEqual(
            results["global"]["final_eps"],
            {0.15: 0.1499994179954894, 3.0: 2.999999783932882, 5.0: 4.999999232325652},
            "Parallelized QC Train (baseline random) simulation has been changed from verified results",
        )


if __name__ == "__main__":
    unittest.main()
