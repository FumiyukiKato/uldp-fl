import copy
from typing import Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    FedTcgaBrca,
)

N_SILO = 6
TRAIN_SIZE_LIST = [248, 156, 164, 129, 129, 40]


def update_args(args):
    updated_args = copy.deepcopy(args)
    updated_args.n_silos = N_SILO
    updated_args.n_silo_per_round = N_SILO
    updated_args.local_batch_size = BATCH_SIZE
    return updated_args


# Because Cox Loss needs multiple data to calcuate loss, so we need to
# build user distribution to make sure each silo and user has 2 data at least.
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
        user_list = np.concatenate((user_list, user_list))
        if n_total < n_users * 2:
            ValueError(
                "The number of users * 2 is larger than the total number of records"
            )
        else:
            # bounded zipf distribution
            x = np.arange(1, n_users + 1)
            weights = x ** (-alpha)
            weights /= weights.sum()
            user_indices_of_data = random_state.choice(
                x, size=n_total - 2 * n_users, replace=True, p=weights
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
            silo_ids = increase_min_count(silo_ids, random_state)
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
            n_users, size=int(np.sum(TRAIN_SIZE_LIST) / 2), replace=True
        )
        cursor = -1
        for silo_id, size in enumerate(TRAIN_SIZE_LIST):
            for _ in range(int(size / 2)):
                cursor += 1
                user_id = random_selected_user_ids[cursor]
                user_ids_per_silo[silo_id].append(user_id)
                user_ids_per_silo[silo_id].append(user_id)
                if user_id not in user_hist_per_silo[silo_id]:
                    user_hist_per_silo[silo_id][user_id] = 0
                user_hist_per_silo[silo_id][user_id] += 2
            if size % 2 == 1:
                user_id = random_selected_user_ids[cursor]
                user_ids_per_silo[silo_id].append(user_id)
                user_hist_per_silo[silo_id][user_id] += 1
    else:
        raise ValueError("Unknown user distribution: {}".format(user_dist))

    user_dist_per_silo = {}
    for silo_id in range(N_SILO):
        random_state.shuffle(user_ids_per_silo[silo_id])
        user_dist_per_silo[silo_id] = (
            user_hist_per_silo[silo_id],
            user_ids_per_silo[silo_id],
        )

    return user_dist_per_silo


def increase_min_count(silo_ids: list, random_state: np.random.RandomState):
    new_silo_ids = []
    ids, counts = np.unique(silo_ids, return_counts=True)
    over_two = ids[counts >= 2]
    under_two = ids[counts < 2]

    if len(over_two) <= 0:
        increased_id = random_state.choice(under_two)
        over_two = np.array([increased_id])
        under_two = np.delete(under_two, np.where(under_two == increased_id))
        remove_id = random_state.choice(under_two)
        under_two = np.delete(under_two, np.where(under_two == remove_id))

    for i in range(len(silo_ids)):
        if silo_ids[i] in under_two:
            selected_id = random_state.choice(over_two)
            new_silo_ids.append(selected_id)
        else:
            new_silo_ids.append(silo_ids[i])
    return new_silo_ids


def custom_load_dataset(
    random_state: np.random.RandomState,
    silo_id: int = None,
    n_users: int = None,
    user_alpha: float = 1.5,
    user_dist: str = "zipf",
) -> Tuple:
    all_train_dataset = FedTcgaBrca(train=True, pooled=True)
    all_test_dataset = FedTcgaBrca(train=False, pooled=True)
    user_dist_per_silo = build_user_dist(
        n_users=n_users,
        random_state=random_state,
        alpha=user_alpha,
        user_dist=user_dist,
    )

    dataset_for_each_silo = {}
    for i in range(N_SILO):
        training_dataset = FedTcgaBrca(center=i, train=True, pooled=False)
        test_dataset = FedTcgaBrca(center=i, train=False, pooled=False)
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
    return torch.optim.SGD(model.parameters(), lr=learning_rate)


def custom_metric():
    return metric


if __name__ == "__main__":
    device = "cpu"
    training_dataset = FedTcgaBrca(train=True, pooled=True, debug=False)
    test_dataset = FedTcgaBrca(train=False, pooled=True, debug=False)
    train_dataloader = DataLoader(
        FedTcgaBrca(train=True, pooled=True, debug=False),
        num_workers=0,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        FedTcgaBrca(train=False, pooled=True, debug=False),
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

    for epoch in range(NUM_EPOCHS_POOLED):
        for idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = lossfunc(outputs, y)
            loss.backward()
            optimizer.step()

    results_dict = {}
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

    # C-index
    print(results_dict)
