import unittest
from pathlib import Path
import os
import numpy as np
import torch
from torchvision import datasets, transforms

import dataset
from dataset import DATA_SET_DIR

path_project = str(Path(__file__).resolve().parent.parent.parent)
seed = 0


class TestDataset(unittest.TestCase):
    def test_shuffle_test_and_train(self):
        dataset_name = "mnist"
        data_dir = os.path.join(path_project, DATA_SET_DIR, dataset_name)
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=apply_transform
        )
        self.assertIsInstance(
            train_dataset.targets[0:10], torch.Tensor, "targets is tensor"
        )
        self.assertListEqual(
            list(train_dataset.targets[0:10]),
            [5, 0, 4, 1, 9, 2, 1, 3, 1, 4],
            "train_dataset is equal to the verified dataset",
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=apply_transform
        )
        self.assertListEqual(
            list(test_dataset.targets[0:10]),
            [7, 2, 1, 0, 4, 1, 4, 9, 5, 9],
            "test_dataset is equal to the verified dataset",
        )

        random_state = np.random.RandomState(seed=seed)
        updated_train_dataset, updated_test_dataset = dataset.shuffle_test_and_train(
            random_state, train_dataset, test_dataset
        )
        self.assertListEqual(
            [label for value, label in updated_train_dataset[0:10]],
            [5, 7, 4, 8, 9, 9, 9, 4, 4, 5],
            "updated_train_dataset is equal to the verified sampled dataset",
        )
        self.assertListEqual(
            [label for value, label in updated_test_dataset[0:10]],
            [6, 2, 6, 4, 0, 2, 6, 7, 9, 0],
            "updated_test_dataset is equal to the verified sampled dataset",
        )

    def test_distribute_data_to_users_with_uniform_iid(self):
        random_state = np.random.RandomState(seed=seed)
        n_data = 60000
        n_users = 1000
        user_indices_of_data = dataset.distribute_data_to_users(
            random_state,
            n_data,
            "uniform-iid",
            n_users,
        )
        indices = np.where(user_indices_of_data == 0)[0]

        # import matplotlib.pyplot as plt
        # plt.hist(user_indices_of_data, bins=np.arange(1, n_users+2))
        # plt.show()

        self.assertListEqual(
            list(indices[0:10]),
            [347, 597, 1037, 1564, 2029, 3095, 4390, 6123, 7608, 8944],
            "user_indices_of_data is equal to the verified sampled dataset",
        )
        self.assertEqual(
            len(indices),
            60,
            "user_indices_of_data is equal to the verified sampled dataset",
        )

        # user_labels = [updated_train_dataset[i][1] for i in indices]
        # n_total_labels = 10
        # plt.hist(user_labels, bins=np.arange(1, n_total_labels+2))
        # plt.show()

    def test_distribute_data_to_users_with_uniform_noniid(self):
        random_state = np.random.RandomState(seed=seed)
        n_data = 60000
        n_users = 1000
        n_labels = 3
        try:
            _ = dataset.distribute_data_to_users(
                random_state,
                n_data,
                "uniform-noniid",
                n_users,
            )
        except AssertionError:
            self.assertTrue(True, "AssertionError is not raised")

        random_state = np.random.RandomState(seed=seed)
        train_dataset = self.prepare_mnist_train_dataset(random_state)
        labels = [label for _, label in train_dataset]
        user_indices_of_data = dataset.distribute_data_to_users(
            random_state,
            n_data,
            "uniform-noniid",
            n_users,
            labels=labels,
            n_labels=n_labels,
        )

        # the data that has 0 as index may have over 4 labels to perserve label number constraint for other data
        indices_1 = np.where(user_indices_of_data == 1)[0]
        indices_2 = np.where(user_indices_of_data == 2)[0]

        self.assertListEqual(
            list(indices_1[0:10]),
            [201, 964, 2535, 3243, 3362, 4898, 6356, 7743, 7847, 8651],
            "user_indices_of_data is equal to the verified sampled dataset",
        )
        self.assertEqual(
            len(indices_1),
            60,
            "user_indices_of_data is equal to the verified sampled dataset",
        )

        unique_user_labels = set([train_dataset[i][1] for i in indices_1])
        self.assertSetEqual(
            unique_user_labels, {8, 9, 7}, "set of unique labels is correct"
        )
        unique_user_labels_2 = set([train_dataset[i][1] for i in indices_2])
        self.assertSetEqual(
            unique_user_labels_2, {1, 4, 7}, "set of unique labels is correct"
        )

    def test_distribute_data_to_users_with_zipf_iid(self):
        random_state = np.random.RandomState(seed=seed)
        n_data = 60000
        n_users = 5000
        alpha = 0.3
        user_indices_of_data = dataset.distribute_data_to_users(
            random_state,
            n_data,
            "zipf-iid",
            n_users,
            alpha=alpha,
        )

        # plt.hist(user_indices_of_data, bins=np.arange(1, n_users + 2))
        # plt.show()

        indices_0 = np.where(user_indices_of_data == 0)[0]
        indices_1 = np.where(user_indices_of_data == 1)[0]
        indices_4999 = np.where(user_indices_of_data == 4999)[0]

        self.assertEqual(
            len(indices_0), 115, "user_indices_of_data is equal to the verified dataset"
        )
        self.assertEqual(
            len(indices_1), 96, "user_indices_of_data is equal to the verified dataset"
        )
        self.assertEqual(
            len(indices_4999),
            12,
            "user_indices_of_data is equal to the verified dataset",
        )

    def test_distribute_data_to_users_with_zipf_noniid(self):
        random_state = np.random.RandomState(seed=seed)
        n_data = 60000
        n_users = 5000
        alpha = 0.3
        n_labels = 3
        train_dataset = self.prepare_mnist_train_dataset(random_state)
        labels = [label for _, label in train_dataset]
        user_indices_of_data = dataset.distribute_data_to_users(
            random_state,
            n_data,
            "zipf-noniid",
            n_users,
            alpha=alpha,
            labels=labels,
            n_labels=n_labels,
        )

        indices_0 = np.where(user_indices_of_data == 0)[0]
        indices_1 = np.where(user_indices_of_data == 1)[0]
        indices_4999 = np.where(user_indices_of_data == 4999)[0]

        self.assertEqual(
            len(indices_0),
            174,
            "user_indices_of_data is equal to the verified sampled dataset",
        )

        self.assertListEqual(
            list(indices_1[0:10]),
            [148, 354, 466, 2187, 2283, 2655, 3196, 3399, 3486, 4100],
            "user_indices_of_data is equal to the verified sampled dataset",
        )
        self.assertEqual(
            len(indices_1),
            84,
            "user_indices_of_data is equal to the verified sampled dataset",
        )
        unique_user_labels = set([train_dataset[i][1] for i in indices_1])
        self.assertSetEqual(
            unique_user_labels, {1, 5, 6}, "set of unique labels is correct"
        )

        self.assertEqual(
            len(indices_4999),
            3,
            "user_indices_of_data is equal to the verified sampled dataset",
        )
        unique_user_labels = set([train_dataset[i][1] for i in indices_4999])
        self.assertSetEqual(unique_user_labels, {2}, "set of unique labels is correct")

    def test_distribute_data_to_silos_with_uniform(self):
        random_state = np.random.RandomState(seed=seed)
        n_data = 60000
        n_silos = 10
        data_indices_per_silos = dataset.distribute_data_to_silos(
            random_state,
            n_data,
            "uniform",
            n_silos,
        )
        silo_0 = data_indices_per_silos[0]
        self.assertListEqual(
            list(silo_0[0:10]),
            [3048, 50959, 12231, 5701, 18952, 2233, 45995, 56925, 45519, 36867],
            "data_indices_per_silos is equal to the verified dataset",
        )
        self.assertEqual(
            len(silo_0),
            6000,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        silo_1 = data_indices_per_silos[1]
        self.assertEqual(
            len(silo_1),
            6000,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )

    def test_distribute_data_to_silos_with_p(self):
        random_state = np.random.RandomState(seed=seed)
        n_data = 60000
        n_silos = 3
        p_list = [0.5, 0.25, 0.25]
        data_indices_per_silos = dataset.distribute_data_to_silos(
            random_state, n_data, "p", n_silos, p_list=p_list
        )
        silo_0 = data_indices_per_silos[0]
        self.assertListEqual(
            list(silo_0[0:10]),
            [4, 6, 9, 14, 15, 16, 22, 24, 26, 29],
            "data_indices_per_silos is equal to the verified dataset",
        )
        self.assertEqual(
            len(silo_0),
            30050,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        silo_1 = data_indices_per_silos[1]
        self.assertEqual(
            len(silo_1),
            15034,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )

    def test_distribute_data_to_silos_with_zipf(self):
        random_state = np.random.RandomState(seed=seed)
        n_data = 60000
        n_silos = 10
        alpha = 0.5
        data_indices_per_silos = dataset.distribute_data_to_silos(
            random_state, n_data, "zipf", n_silos, alpha=alpha
        )
        silo_0 = data_indices_per_silos[0]
        self.assertListEqual(
            list(silo_0[0:10]),
            [14, 15, 16, 24, 26, 34, 43, 47, 53, 55],
            "data_indices_per_silos is equal to the verified dataset",
        )
        self.assertEqual(
            len(silo_0),
            12156,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        silo_1 = data_indices_per_silos[1]
        self.assertEqual(
            len(silo_1),
            8392,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        silo_9 = data_indices_per_silos[9]
        self.assertEqual(
            len(silo_9),
            3684,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        self.assertEqual(
            sum([len(lst) for lst in data_indices_per_silos.values()]),
            60000,
            "The sum of silos is 60000",
        )

    def test_distribute_data_to_silos_with_user_silo_matrix(self):
        random_state = np.random.RandomState(seed=seed)
        n_data = 60000
        n_silos = 4
        n_users = 1000
        alpha = 0.8
        user_indices_of_data = dataset.distribute_data_to_users(
            random_state,
            n_data,
            "zipf-iid",
            n_users,
            alpha=alpha,
        )
        user_silo_matrix_1 = [[0.5, 0.2, 0.2, 0.1] for u in range(500)]
        user_silo_matrix_2 = [[0.1, 0.5, 0.3, 0.1] for u in range(500)]
        user_silo_matrix = np.array(user_silo_matrix_1 + user_silo_matrix_2)

        data_indices_per_silos = dataset.distribute_data_to_silos(
            random_state,
            n_data,
            "user-silo-matrix",
            n_silos,
            n_users=n_users,
            data_indices_of_users=user_indices_of_data,
            user_silo_matrix=user_silo_matrix,
        )
        silo_0 = data_indices_per_silos[0]
        self.assertListEqual(
            list(silo_0[0:10]),
            [43, 99, 171, 179, 198, 204, 214, 233, 262, 268],
            "data_indices_per_silos is equal to the verified dataset",
        )
        self.assertEqual(
            len(silo_0),
            25973,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        all_user_0_set = set(np.where(user_indices_of_data == 0)[0])
        user_0_set_of_silo_0 = set(silo_0)
        ratio = len(all_user_0_set.intersection(user_0_set_of_silo_0)) / len(
            all_user_0_set
        )
        self.assertTrue(
            0.48 <= ratio and ratio <= 0.52,
            "The ratio of user 0 in silo 0 over all user 0 data is approximately equal to user_silo_matrix",
        )

        silo_1 = data_indices_per_silos[1]
        all_user_5_set = set(np.where(user_indices_of_data == 5)[0])
        user_5_set_of_silo_1 = set(silo_1)
        ratio = len(all_user_5_set.intersection(user_5_set_of_silo_1)) / len(
            all_user_5_set
        )
        self.assertTrue(
            0.18 <= ratio and ratio <= 0.22,
            "The ratio of user 5 in silo 1 over all user 5 data is approximately equal to user_silo_matrix",
        )

        silo_1 = data_indices_per_silos[1]
        self.assertEqual(
            len(silo_1),
            14783,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        silo_2 = data_indices_per_silos[2]
        self.assertEqual(
            len(silo_2),
            13089,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        silo_3 = data_indices_per_silos[3]
        self.assertEqual(
            len(silo_3),
            6155,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )

    def prepare_mnist_train_dataset(self, random_state: np.random.RandomState):
        dataset_name = "mnist"
        data_dir = os.path.join(path_project, DATA_SET_DIR, dataset_name)
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=apply_transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=apply_transform
        )
        updated_train_dataset, _ = dataset.shuffle_test_and_train(
            random_state, train_dataset, test_dataset
        )
        return updated_train_dataset

    def test_something(self):
        # テストケースを記述
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
