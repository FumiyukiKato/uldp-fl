import unittest
from pathlib import Path
import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

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
            random_state, n_data, "uniform", n_silos, n_users=100
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

    def test_distribute_data_to_silos_with_zipf(self):
        random_state = np.random.RandomState(seed=seed)
        n_data = 60000
        n_silos = 10
        alpha = 0.5
        n_users = 100
        data_indices_of_users = dataset.distribute_data_to_users(
            random_state,
            n_data,
            "uniform-iid",
            n_users,
        )
        data_indices_per_silos = dataset.distribute_data_to_silos(
            random_state,
            n_data,
            "zipf",
            n_silos,
            alpha=alpha,
            n_users=n_users,
            data_indices_of_users=data_indices_of_users,
        )
        silo_0 = data_indices_per_silos[0]
        self.assertListEqual(
            list(silo_0[0:10]),
            [6079, 37093, 52535, 1564, 15276, 597, 35652, 29977, 48767, 8991],
            "data_indices_per_silos is equal to the verified dataset",
        )
        self.assertEqual(
            len(silo_0),
            5731,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        silo_1 = data_indices_per_silos[1]
        self.assertEqual(
            len(silo_1),
            5880,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        silo_9 = data_indices_per_silos[9]
        self.assertEqual(
            len(silo_9),
            6212,
            "The length of data_indices_per_silos is equal to the verified dataset",
        )
        self.assertEqual(
            sum([len(lst) for lst in data_indices_per_silos.values()]),
            60000,
            "The sum of silos is 60000",
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


if __name__ == "__main__":
    unittest.main()
