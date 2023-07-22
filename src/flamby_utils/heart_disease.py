import copy
from typing import Tuple
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    metric,
)


N_SILO = 4
TRAIN_SIZE_LIST = [199, 172, 30, 85]


def update_args(args):
    updated_args = copy.deepcopy(args)
    updated_args.n_silos = N_SILO
    updated_args.n_silo_per_round = N_SILO
    updated_args.local_batch_size = BATCH_SIZE
    return updated_args


def build_user_dist(
    n_users: int,
    random_state: np.random.RandomState,
    alpha: float = 1.5,
    user_dist: str = "zipf",
):
    if user_dist.startswith("zipf"):
        # Ensure that every user has at least one record
        # Other data is allocated according to the zipf distribution
        n_total = np.sum(TRAIN_SIZE_LIST)

        user_list = np.arange(n_users)
        if n_total < n_users:
            user_id_of_records = np.arange(n_total)
        else:
            # bounded zipf distribution
            x = np.arange(1, n_users + 1)
            weights = x ** (-alpha)
            weights /= weights.sum()
            user_indices_of_data = random_state.choice(
                x, size=n_total - n_users, replace=True, p=weights
            )
            user_indices_of_data = user_indices_of_data - 1
            user_id_of_records = np.concatenate([user_list, user_indices_of_data])
        random_state.shuffle(user_id_of_records)
        user_id_of_records = user_id_of_records.tolist()
        _, count_per_user = np.unique(user_id_of_records, return_counts=True)

        user_ids_per_silo = {i: [] for i in range(N_SILO)}
        user_hist_per_silo = {i: {} for i in range(N_SILO)}
        ratios_per_silo = {}
        MAIN_RATIO = 0.8

        for silo_id in range(N_SILO):
            base_ratios = [(1.0 - MAIN_RATIO) / (N_SILO - 1)] * N_SILO
            base_ratios[silo_id] = MAIN_RATIO
            ratios_per_silo[silo_id] = base_ratios

        for user_id in range(n_users):
            count = count_per_user[user_id]
            selected_silo = random_state.choice(N_SILO)
            silo_ids = random_state.choice(
                N_SILO, size=count, replace=True, p=ratios_per_silo[selected_silo]
            )
            for silo_id in silo_ids:
                # if the number of records in the silo is larger than the limit, choose another silo
                while len(user_ids_per_silo[silo_id]) >= TRAIN_SIZE_LIST[silo_id]:
                    silo_id = (silo_id + 1) % N_SILO
                if user_id not in user_hist_per_silo[silo_id]:
                    user_hist_per_silo[silo_id][user_id] = 0

                user_ids_per_silo[silo_id].append(user_id)
                user_hist_per_silo[silo_id][user_id] += 1

    elif user_dist.startswith("uniform"):
        user_ids_per_silo = {}
        user_hist_per_silo = {}
        for silo_id in range(N_SILO):
            user_ids_per_silo[silo_id] = []
            user_hist_per_silo[silo_id] = {}
        random_selected_user_ids = random_state.choice(
            n_users, size=np.sum(TRAIN_SIZE_LIST), replace=True
        )
        cursor = 0
        for silo_id, size in enumerate(TRAIN_SIZE_LIST):
            for _ in range(size):
                user_id = random_selected_user_ids[cursor]
                cursor += 1
                user_ids_per_silo[silo_id].append(user_id)
                if user_id not in user_hist_per_silo[silo_id]:
                    user_hist_per_silo[silo_id][user_id] = 0
                user_hist_per_silo[silo_id][user_id] += 1

    user_dist_per_silo = {}
    for silo_id in range(N_SILO):
        random_state.shuffle(user_ids_per_silo[silo_id])
        user_dist_per_silo[silo_id] = (
            user_hist_per_silo[silo_id],
            user_ids_per_silo[silo_id],
        )

    return user_dist_per_silo


def custom_load_dataset(
    random_state: np.random.RandomState,
    silo_id: int = None,
    n_users: int = None,
    user_alpha: float = 1.5,
    user_dist: str = "zipf",
) -> Tuple:
    all_train_dataset = FedHeartDisease(train=True, pooled=True, debug=False)
    all_test_dataset = FedHeartDisease(train=False, pooled=True, debug=False)
    user_dist_per_silo = build_user_dist(
        n_users=n_users,
        random_state=random_state,
        alpha=user_alpha,
        user_dist=user_dist,
    )

    dataset_for_each_silo = {}
    for i in range(N_SILO):
        training_dataset = FedHeartDisease(
            center=i, train=True, pooled=False, debug=False
        )
        test_dataset = FedHeartDisease(center=i, train=False, pooled=False, debug=False)
        user_hist, user_ids = user_dist_per_silo[i]
        dataset_for_each_silo[i] = (training_dataset, test_dataset, user_hist, user_ids)

    if silo_id is not None:
        return dataset_for_each_silo[silo_id]
    return all_train_dataset, all_test_dataset, dataset_for_each_silo


def custom_model():
    return Baseline()


def custom_loss():
    return BaselineLoss()


def custom_optimizer(model, learning_rate: float = LR):
    return optim.SGD(model.parameters(), lr=learning_rate)
    # return optim.Adam(model.parameters(), lr=LR)


def custom_metric():
    return metric


if __name__ == "__main__":
    device = "cpu"
    training_dataset = FedHeartDisease(train=True, pooled=True, debug=False)
    test_dataset = FedHeartDisease(train=False, pooled=True, debug=False)
    train_dataloader = DataLoader(
        FedHeartDisease(train=True, pooled=True, debug=False),
        num_workers=0,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        FedHeartDisease(train=False, pooled=True, debug=False),
        num_workers=0,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    seed = 0
    torch.manual_seed(seed)
    model = custom_model()
    model = model.to(device)
    lossfunc = custom_loss()
    optimizer = custom_optimizer(model)

    for e in tqdm(range(NUM_EPOCHS_POOLED)):
        for s, (X, y) in enumerate(train_dataloader):
            # traditional training loop with optional GPU transfer
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            lm = lossfunc(y_pred, y)
            lm.backward()
            optimizer.step()

    results_dict = {}
    y_true_dict = {}
    y_pred_dict = {}

    model.eval()
    with torch.no_grad():
        y_pred_final = []
        y_true_final = []
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X).detach().cpu()
            y = y.detach().cpu()
            y_pred_final.append(y_pred.numpy())
            y_true_final.append(y.numpy())

        y_true_final = np.concatenate(y_true_final)
        y_pred_final = np.concatenate(y_pred_final)
        results_dict["test"] = metric(y_true_final, y_pred_final)

    print("Benchmark Results on Heart Disease pooled:")
    print(results_dict)
