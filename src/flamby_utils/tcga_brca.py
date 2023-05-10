import copy
from typing import Dict, Tuple
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

N_SILO = 5
TRAIN_SIZE_LIST = [248, 156, 164, 129, 129, 40]


def update_args(args):
    updated_args = copy.deepcopy(args)
    updated_args.n_silos = N_SILO
    updated_args.n_silo_per_round = N_SILO
    updated_args.local_batch_size = BATCH_SIZE
    updated_args.learning_rate = LR
    return updated_args


# とりあえず適当に作った
def build_user_dist(
    all_training_dataset: FedTcgaBrca,
    n_users: int = 500,
) -> Dict:
    n_all_dataset = len(all_training_dataset)
    user_id_of_records = np.random.RandomState(seed=0).choice(
        n_users, size=n_all_dataset, replace=True
    )
    user_ids_per_silo = {}
    user_hist_per_silo = {}
    record_id = 0
    for silo_id, silo_size in enumerate(TRAIN_SIZE_LIST):
        user_ids_per_silo[silo_id] = []
        user_hist_per_silo[silo_id] = {}
        for _ in range(silo_size):
            user_id = user_id_of_records[record_id]
            user_ids_per_silo[silo_id].append(user_id)
            if user_id not in user_hist_per_silo[silo_id]:
                user_hist_per_silo[silo_id][user_id] = 0
            user_hist_per_silo[silo_id][user_id] += 1

    user_dist = {}
    for silo_id in range(N_SILO):
        user_dist[silo_id] = (user_hist_per_silo[silo_id], user_ids_per_silo[silo_id])

    return user_dist


def custom_load_dataset(silo_id: int = None) -> Tuple:
    all_training_dataset = FedTcgaBrca(train=True, pooled=True)
    all_test_dataset = FedTcgaBrca(train=False, pooled=True)
    user_dist_per_silo = build_user_dist(all_training_dataset)

    dataset_for_each_silo = {}
    for i in range(N_SILO):
        training_dataset = FedTcgaBrca(center=i, train=True, pooled=False)
        test_dataset = FedTcgaBrca(center=i, train=False, pooled=False)
        user_hist, user_ids = user_dist_per_silo[i]
        dataset_for_each_silo[i] = (training_dataset, test_dataset, user_hist, user_ids)

    if silo_id is not None:
        return dataset_for_each_silo[silo_id]
    return all_training_dataset, all_test_dataset, dataset_for_each_silo


def custom_model():
    return Baseline()


def custom_loss():
    return BaselineLoss()


def custom_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=LR)


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
