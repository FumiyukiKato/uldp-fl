import copy
import os
import numpy as np
from typing import Tuple, List, Dict
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
import torch

from mylogger import logger

DATA_SET_DIR = "dataset"
LOCAL_TEST_RATIO = 0.1


def shuffle_test_and_train(
    random_state: np.random.RandomState,
    train_dataset: datasets,
    test_dataset: datasets,
) -> Tuple[List[Tuple[torch.Tensor, int]], List[Tuple[torch.Tensor, int]]]:
    """
    Shuffle the test and train datasets and split them into train and test sets.

    Inputs:
        random_state(np.random.RandomState): random seed
        train_dataset(datasets): training dataset
        test_dataset(datasets): testing dataset
    Return: updated training and testing datasets
    """

    combined_data = ConcatDataset([train_dataset, test_dataset])
    targets = torch.cat((train_dataset.targets, test_dataset.targets))
    test_ratio = len(test_dataset) / (len(test_dataset) + len(train_dataset))

    updated_train_dataset, updated_test_dataset = train_test_split(
        combined_data,
        test_size=test_ratio,
        random_state=random_state,
        stratify=targets,
    )

    return updated_train_dataset, updated_test_dataset


def divide_dataset(
    random_state: np.random.RandomState,
    train_dataset: list,
    labels: np.ndarray,
    n_users: int,
    n_silos: int,
    user_dist: str,
    silo_dist: str,
    user_alpha: float = None,
    silo_alpha: float = None,
    p_list: list[float] = None,
    n_labels: int = None,
    user_silo_matrix: np.ndarray = None,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    The relationship between users and silos can be uniformly distributed or follow a zipf distribution.
    Users and data are assumed to be dependent, but silos and data are assumed to be independent.

    Inputs:
        random_state(np.random.RandomState): random state
        train_dataset(list): training dataset
        labels(np.ndarray): labels of the training dataset
        n_users(int): number of users
        n_silos(int): number of silos
        user_dist(str): distribution of users to silos, can be "uniform" or "zipf" (need to specify alpha)
        silo_dist(str): distribution of data to silos, can be "uniform" or "zipf" (need to specify alpha) or "p" (need to specify p_list)
        user_alpha(float): distribution parameter for zipf distribution of users
        silo_alpha(float): distribution parameter for zipf distribution of silos
        p_list(list[float]): list of probabilities for each silo
        n_labels(int): number of distinct labels
        user_silo_matrix(np.ndarray): matrix of user-silo relationship
    Return(dict[int, np.ndarray]): a dictionary of data indices per silo.
    """
    n_data = len(train_dataset)

    # First, allocate users to data
    data_indices_of_users = distribute_data_to_users(
        random_state, n_data, user_dist, n_users, user_alpha, labels, n_labels
    )

    # Second, allocate data to silos (indepedent of users distribution)
    data_indices_per_silos = distribute_data_to_silos(
        random_state,
        n_data,
        silo_dist,
        n_silos,
        p_list=p_list,
        alpha=silo_alpha,
        n_users=n_users,
        data_indices_of_users=data_indices_of_users,
        user_silo_matrix=user_silo_matrix,
    )

    return data_indices_per_silos, data_indices_of_users


def distribute_data_to_silos(
    random_state: np.random.RandomState,
    n_data: int,
    dist: str,
    n_silos: int,
    p_list: int = None,
    alpha: float = None,
    n_users: int = None,
    data_indices_of_users: np.ndarray = None,
    user_silo_matrix: np.ndarray = None,
) -> Dict[int, np.ndarray]:
    """
    Distribute data to silos.
    User and silo is independent except dist = "user-silo-matrix".

    Inputs:
        random_state: random state
        n_data: number of data
        dist: distribution of data to silos
        n_silos: number of silos
        p_list: list of probabilities for each silo
        alpha: alpha parameter for zipf distribution
    Return: a dictionary of data indices per silo.
    """
    data_indices_per_silos = {i: [] for i in range(n_silos)}

    if dist == "uniform":
        samples = random_state.permutation(np.arange(n_data))
        for i in range(n_silos):
            data_indices_per_silos[i] = samples[i::n_silos]
    elif dist == "p":
        assert len(p_list) == n_silos, "p_list must have the same length as n_silos"
        samples = random_state.choice(n_silos, size=n_data, replace=True, p=p_list)
        for idx, i in enumerate(samples):
            data_indices_per_silos[i].append(idx)
    elif dist == "zipf":
        # bounded zipf distribution
        N = n_silos
        x = np.arange(1, N + 1)
        weights = x ** (-alpha)
        weights /= weights.sum()
        silo_indices_of_data = random_state.choice(
            x, size=n_data, replace=True, p=weights
        )
        silo_indices_of_data = silo_indices_of_data - 1
        for idx, i in enumerate(silo_indices_of_data):
            data_indices_per_silos[i].append(idx)
    elif dist == "user-silo-matrix":
        assert user_silo_matrix is not None, "user_silo_matrix must be provided"
        indices_per_user = {i: [] for i in range(n_users)}
        for idx, user_id in enumerate(data_indices_of_users):
            indices_per_user[user_id].append(idx)

        # sampling silos for each user based on user_silo_matrix
        for user_id, indices in indices_per_user.items():
            selected_silo_ids = random_state.choice(
                range(n_silos),
                size=len(indices),
                p=user_silo_matrix[user_id],
                replace=True,
            )
            for idx, silo_id in zip(indices, selected_silo_ids):
                data_indices_per_silos[silo_id].append(idx)
    else:
        raise ValueError("dist must be either uniform, p, or zipf.")
    return data_indices_per_silos


def distribute_data_to_users(
    random_state: np.random.RandomState,
    n_data: int,
    dist: str,
    n_users: int,
    alpha: float = None,
    labels: np.ndarray = None,
    n_labels: int = None,
) -> np.ndarray:
    """
    Distribute data to users.

    Inputs:
        random_state(np.random.RandomState): random state
        n_data(int): number of data points, i.e. len(train_dataset)
        dist(str): distribution of data to users
        n_users(int): number of users
        alpha(float): alpha parameter for zipf distribution
        labels(np.ndarray): labels of data points
        n_labels(int): number of distinct labels
    Return(np.ndarray): user indices of data points
    """
    if dist.startswith("uniform-"):
        user_indices_of_data = random_state.permutation(
            int(np.ceil(n_data / n_users)) * list(range(n_users))
        )
    elif dist.startswith("zipf-"):
        # bounded zipf distribution
        N = n_users
        x = np.arange(1, N + 1)
        weights = x ** (-alpha)
        weights /= weights.sum()
        user_indices_of_data = random_state.choice(
            x, size=n_data, replace=True, p=weights
        )
        user_indices_of_data = user_indices_of_data - 1
    else:
        raise ValueError("dist must be either uniform-* or zipf-*.")

    if dist.endswith("-noniid"):
        assert (
            labels is not None
        ), "labels must be provided for non-iid data distribution"
        assert (
            n_labels is not None
        ), "n_labels must be provided for non-iid data distribution"
        user_indices_of_data = allocate_data_noniid(
            random_state, user_indices_of_data, n_data, labels, n_labels
        )

    return user_indices_of_data


def allocate_data_noniid(
    random_state: np.random.RandomState,
    user_indices_of_data: np.ndarray,
    n_data: int,
    labels: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """
    Distribute non-iid data to users.

    Inputs:
        random_state(np.random.RandomState): random state
        user_indices_of_data(np.ndarray): user indices of data points
        n_data(int): number of data points, i.e. len(train_dataset)
        labels(np.ndarray): labels of data
        n_labels(int): number of unique labels per user
    Return(np.ndarray): user indices of data points
    """
    unique_users, counts = np.unique(user_indices_of_data, return_counts=True)
    n_data_per_user = dict(zip(unique_users, counts))

    unique_labels = np.unique(labels)

    # sort data indices by labels
    idxs = np.arange(n_data)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide indices by label
    cursor_per_label = {}
    idxs_per_label = {}
    last_idx = 0
    last_label = unique_labels[0]
    for label in unique_labels[1:]:
        first_idx = np.where(idxs_labels[1, :] == label)[0][0]
        idxs_per_label[last_label] = idxs[last_idx:first_idx]
        cursor_per_label[last_label] = 0
        last_label = label
        last_idx = first_idx
    idxs_per_label[last_label] = idxs[last_idx:]
    cursor_per_label[last_label] = 0

    # shuffle for each label's indices
    for value in idxs_per_label.values():
        random_state.shuffle(value)

    # allocate data to users
    # Note: all of left data will be allocated to user 0
    updated_user_indices_of_data = np.zeros(n_data, dtype=int)
    for user_id, n_data_of_user in n_data_per_user.items():
        if len(unique_labels) > n_labels:
            rand_set = set(random_state.choice(unique_labels, n_labels, replace=False))
        else:
            rand_set = set(unique_labels)
        n_data_of_user_of_label = int(np.round(n_data_of_user / len(rand_set)))
        for rand_label in rand_set:
            selected_data = idxs_per_label[rand_label][
                cursor_per_label[rand_label] : cursor_per_label[rand_label]
                + n_data_of_user_of_label
            ]
            # Corner scenaio: data is not enough for this user and this label
            if len(selected_data) < n_data_of_user_of_label:
                # data is over half of the data
                if len(selected_data) > n_data_of_user_of_label * 0.5:
                    updated_user_indices_of_data[selected_data] = user_id
                    unique_labels = unique_labels[unique_labels != rand_label]
                    logger.debug(
                        f"Minor Warning: data is not enough ({n_data_of_user_of_label} - {len(selected_data)}) for user {user_id} and label {rand_label}."
                    )
                else:  # if data is less than half of the data, then retry
                    done = False
                    for (
                        label
                    ) in (
                        unique_labels
                    ):  # retry for all labels if the number of labels are enough
                        selected_data_retry = idxs_per_label[label][
                            cursor_per_label[label] : cursor_per_label[label]
                            + n_data_of_user_of_label
                        ]
                        if len(selected_data_retry) == n_data_of_user_of_label:
                            updated_user_indices_of_data[selected_data_retry] = user_id
                            cursor_per_label[label] += n_data_of_user_of_label
                            done = True
                            break
                        elif len(selected_data_retry) > n_data_of_user_of_label * 0.5:
                            updated_user_indices_of_data[selected_data_retry] = user_id
                            unique_labels = unique_labels[unique_labels != label]
                            done = True
                            logger.debug(
                                f"Minor Warning: data is not enough ({n_data_of_user_of_label} - {len(selected_data_retry)}) for user {user_id} and label {label}."
                            )
                            break
                    if (
                        not done
                    ):  # if all labels are not enough, first label is selected and used even though it is not enough
                        updated_user_indices_of_data[selected_data] = user_id
                        unique_labels = unique_labels[unique_labels != rand_label]
                        logger.debug(
                            f"Minor Warning: data is not enough ({n_data_of_user_of_label} - {len(selected_data)}) for user {user_id} and label {rand_label}."
                        )
            else:  # Normal scenaio: if the number of data is enough
                updated_user_indices_of_data[selected_data] = user_id
                cursor_per_label[rand_label] += n_data_of_user_of_label
    logger.debug(
        f"Note: index-0 data has {len(np.where(updated_user_indices_of_data == 0)[0])} (If following distribution = {n_data_per_user[0]})."
    )
    return updated_user_indices_of_data


def prepare_silo_dataset(
    train_dataset,
    data_indices_per_silos,
    silo_id,
    silo_random_state,
) -> Tuple[List[Tuple[torch.Tensor, int]], List[Tuple[torch.Tensor, int]], List]:
    silo_indices = data_indices_per_silos[silo_id]
    local_dataset = [train_dataset[i] for i in silo_indices]
    targets = [target for _, target in local_dataset]
    if len(local_dataset) <= 200:
        local_train_dataset, local_test_dataset, local_train_indices = (
            local_dataset,
            [],
            silo_indices,
        )
    else:
        (
            local_train_dataset,
            local_test_dataset,
            local_train_indices,
            _,
        ) = train_test_split(
            local_dataset,
            silo_indices,
            test_size=LOCAL_TEST_RATIO,
            random_state=silo_random_state,
            stratify=targets,
        )

    return local_train_dataset, local_test_dataset, local_train_indices


def build_user_histogram(local_train_indices, data_indices_of_users) -> Dict[int, int]:
    user_histogram = {}
    user_ids_of_local_train_dataset = []
    for idx in local_train_indices:
        user_id = data_indices_of_users[idx]
        if user_id not in user_histogram:
            user_histogram[user_id] = 1
        else:
            user_histogram[user_id] += 1
        user_ids_of_local_train_dataset.append(user_id)
    return user_histogram, user_ids_of_local_train_dataset


def load_pre_seperated_dataset(
    dataset_name: str,
    random_state: np.random.RandomState,
    silo_id: int = None,
    n_users: int = None,
) -> Tuple:
    if dataset_name == "heart_disease":
        from flamby_utils import heart_disease

        dataset = heart_disease.custom_load_dataset(
            random_state=random_state, silo_id=silo_id, n_users=n_users
        )
    elif dataset_name == "isic":
        from flamby_utils import isic

        dataset = isic.custom_load_dataset(
            random_state=random_state, silo_id=silo_id, n_users=n_users
        )
    elif dataset_name == "tcga_brca":
        from flamby_utils import tcga_brca

        dataset = tcga_brca.custom_load_dataset(
            random_state=random_state, silo_id=silo_id, n_users=n_users
        )
    else:
        raise ValueError("Invalid dataset name.")
    return dataset


def load_dataset(
    random_state: np.random.RandomState,
    dataset_name: str,
    path_project: str,
    n_users: int,
    n_silos: int,
    user_dist: str,
    silo_dist: str,
    user_alpha: float = None,
    silo_alpha: float = None,
    p_list: list[float] = None,
    n_labels: int = None,
    user_silo_matrix: np.ndarray = None,
    silo_id: int = None,
    is_simulation: bool = False,
) -> Tuple[List[Tuple[torch.Tensor, int]], List[Tuple[torch.Tensor, int]]]:
    logger.info("Start prepare dataset...")
    if dataset_name in ["heart_disease", "isic", "tcga_brca"]:
        # for simulator
        if is_simulation:
            return load_pre_seperated_dataset(
                dataset_name, random_state, n_users=n_users
            )
        # for silo
        if silo_id is not None:
            return load_pre_seperated_dataset(
                dataset_name, random_state, silo_id=silo_id, n_users=n_users
            )
        # for server
        all_training_dataset, all_test_dataset, _ = load_pre_seperated_dataset(
            dataset_name, random_state, n_users=n_users
        )
        return all_training_dataset, all_test_dataset

    if dataset_name == "cifar10":
        data_dir = os.path.join(path_project, DATA_SET_DIR, "cifar10")
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=apply_transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=apply_transform
        )
        train_dataset.targets = torch.tensor(train_dataset.targets)
        test_dataset.targets = torch.tensor(test_dataset.targets)
        labels = train_dataset.targets.numpy()

    elif dataset_name == "mnist":
        data_dir = os.path.join(path_project, DATA_SET_DIR, "mnist")
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
        labels = train_dataset.targets.numpy()

    elif dataset_name == "cifar100":
        # For simplicity, we use statistics of the first traindaset to the first testdaset though the both dataset are mixed in later
        data_dir = os.path.join(path_project, DATA_SET_DIR, "cifar100")
        CIFAR100_TRAIN_MEAN = (
            0.5070751592371323,
            0.48654887331495095,
            0.4409178433670343,
        )
        CIFAR100_TRAIN_STD = (
            0.2673342858792401,
            0.2564384629170883,
            0.27615047132568404,
        )
        transform_train = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
            ]
        )
        train_dataset = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=transform_test
        )
        train_dataset.targets = torch.tensor(train_dataset.targets)
        test_dataset.targets = torch.tensor(test_dataset.targets)
        labels = train_dataset.targets.numpy()
    else:
        raise ValueError("Dataset not supported")

    train_dataset, test_dataset = shuffle_test_and_train(
        random_state, train_dataset, test_dataset
    )

    data_indices_per_silos, data_indices_of_users = divide_dataset(
        random_state,
        train_dataset,
        labels,
        n_users,
        n_silos,
        user_dist,
        silo_dist,
        user_alpha=user_alpha,
        silo_alpha=silo_alpha,
        p_list=p_list,
        n_labels=n_labels,
        user_silo_matrix=user_silo_matrix,
    )

    # statistics of dataset
    bin_count = sorted(np.bincount(data_indices_of_users))
    logger.info(
        "ALL dataset Percentile of #User's record 0%: {}, 25%: {}, 50%: {}, 75%: {}, 100%: {}".format(
            bin_count[int(len(bin_count) - 1)],
            bin_count[int(len(bin_count) * 0.75 - 1)],
            bin_count[int(len(bin_count) * 0.5 - 1)],
            bin_count[int(len(bin_count) * 0.25 - 1)],
            bin_count[0],
        )
    )

    # for simulator
    if is_simulation:
        dataset_per_silos = {}
        for silo_id in range(n_silos):
            silo_random_state = copy.deepcopy(random_state)
            (
                local_train_dataset,
                local_test_dataset,
                local_train_indices,
            ) = prepare_silo_dataset(
                train_dataset,
                data_indices_per_silos,
                silo_id,
                silo_random_state,
            )

            user_hist, user_ids = build_user_histogram(
                local_train_indices, data_indices_of_users
            )
            dataset_per_silos[silo_id] = (
                local_train_dataset,
                local_test_dataset,
                user_hist,
                user_ids,
            )

            # statistics of each silo's dataset
            logger.info(
                "Silo id: %d, #records = %d, #users = %d",
                silo_id,
                len(local_train_dataset),
                len(set(user_ids)),
            )
            bin_count = np.bincount(list(user_ids) + [0, n_users - 1])
            bin_count[0] = bin_count[0] - 1
            bin_count[n_users - 1] = bin_count[n_users - 1] - 1
            bin_count = sorted(bin_count)
            logger.info(
                "Percentile of #User's record 0%: {}, 25%: {}, 50%: {}, 75%: {}, 100%: {}".format(
                    bin_count[int(len(bin_count) - 1)],
                    bin_count[int(len(bin_count) * 0.75 - 1)],
                    bin_count[int(len(bin_count) * 0.5 - 1)],
                    bin_count[int(len(bin_count) * 0.25 - 1)],
                    bin_count[0],
                )
            )
        return train_dataset, test_dataset, dataset_per_silos
    # for silo
    if silo_id is not None:
        (
            local_train_dataset,
            local_test_dataset,
            local_train_indices,
        ) = prepare_silo_dataset(
            train_dataset,
            data_indices_per_silos,
            silo_id,
            silo_random_state,
        )
        user_hist, user_ids = build_user_histogram(
            local_train_indices, data_indices_of_users
        )
        return local_train_dataset, local_test_dataset, user_hist, user_ids

    # for server
    return train_dataset, test_dataset
