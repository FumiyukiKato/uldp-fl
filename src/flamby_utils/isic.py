import copy
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import metrics

from flamby.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedIsic2019,
    metric,
)

N_SILO = 6


def update_args(args):
    updated_args = copy.deepcopy(args)
    updated_args.n_silos = N_SILO
    updated_args.n_silo_per_round = N_SILO
    updated_args.local_batch_size = BATCH_SIZE
    return updated_args


def build_user_dist(
    all_train_dataset: FedIsic2019,
    n_users: int = 5000,
    random_state: np.random.RandomState = np.random.RandomState(seed=0),
) -> Dict:
    n_train_dataset = len(all_train_dataset)
    user_list = np.arange(n_users)
    if n_train_dataset < n_users:
        user_id_of_records = np.arange(n_train_dataset)
    else:
        user_id_of_records = np.concatenate(
            [
                user_list,
                random_state.choice(user_list, n_train_dataset - len(user_list)),
            ]
        )
    random_state.shuffle(user_id_of_records)
    user_id_of_records = user_id_of_records.tolist()

    user_ids_per_silo = {}
    user_hist_per_silo = {}
    for record_id, user_id in enumerate(user_id_of_records):
        if record_id < len(all_train_dataset):
            silo_id = all_train_dataset.centers[record_id]

            if silo_id not in user_ids_per_silo:
                user_ids_per_silo[silo_id] = []
                user_hist_per_silo[silo_id] = {}
            if user_id not in user_hist_per_silo[silo_id]:
                user_hist_per_silo[silo_id][user_id] = 0
            user_ids_per_silo[silo_id].append(user_id)
            user_hist_per_silo[silo_id][user_id] += 1

    user_dist = {}
    for silo_id in range(N_SILO):
        user_dist[silo_id] = (user_hist_per_silo[silo_id], user_ids_per_silo[silo_id])

    return user_dist


def custom_load_dataset(
    random_state: np.random.RandomState,
    silo_id: int = None,
    n_users: int = None,
    user_alpha: float = 0.5,
    user_dist: str = "uniform",
) -> Tuple:
    all_train_dataset = FedIsic2019(train=True, pooled=True, debug=False)
    all_test_dataset = FedIsic2019(train=False, pooled=True, debug=False)
    user_dist_per_silo = build_user_dist(
        all_train_dataset, random_state=random_state, n_users=n_users
    )

    dataset_for_each_silo = {}
    for i in range(N_SILO):
        training_dataset = FedIsic2019(center=i, train=True, pooled=False, debug=False)
        test_dataset = FedIsic2019(center=i, train=False, pooled=False, debug=False)
        user_hist, user_ids = user_dist_per_silo[i]
        dataset_for_each_silo[i] = (training_dataset, test_dataset, user_hist, user_ids)

    if silo_id is not None:
        return dataset_for_each_silo[silo_id]
    return all_train_dataset, all_test_dataset, dataset_for_each_silo


def custom_model():
    return Baseline()


def custom_loss(train_dataset, device):
    weights = [1] * 8
    for x in train_dataset:
        weights[int(x[1])] += 1
    N = len(train_dataset)
    class_weights = torch.FloatTensor([N / weights[i] for i in range(8)]).to(device)
    return BaselineLoss(alpha=class_weights)


def custom_optimizer(model, learning_rate: float = LR):
    optimize_final_layer_only = False
    if optimize_final_layer_only:
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.base_model._fc.parameters():
            param.requires_grad = True
        optimizer = torch.optim.SGD(model.base_model._fc.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(model.base_model._fc.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.base_model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(model.base_model.parameters(), lr=learning_rate)
    return optimizer


def custom_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5
    )
    return scheduler


def custom_metric():
    return metric


if __name__ == "__main__":
    device = "cpu"
    train_dataset = FedIsic2019(train=True, pooled=True, debug=False)
    test_dataset = FedIsic2019(train=False, pooled=True, debug=False)
    train_dataloader = DataLoader(
        FedIsic2019(train=True, pooled=True, debug=False),
        num_workers=0,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        FedIsic2019(train=False, pooled=True, debug=False),
        num_workers=0,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )

    seed = 0
    torch.manual_seed(seed)
    model = Baseline()
    model = model.to(device)
    lossfunc = custom_loss(train_dataset)
    optimizer = custom_optimizer(model)
    scheduler = custom_scheduler(optimizer)

    for epoch in range(NUM_EPOCHS_POOLED):
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS_POOLED - 1))
        print("-" * 10)

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossfunc(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    results_dict = {}
    y_true_dict = {}
    y_pred_dict = {}
    running_loss = 0.0
    running_corrects = 0
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_true.append(labels)
            outputs = model(inputs)
            loss = lossfunc(outputs, labels)
            _, preds = torch.max(outputs, 1)
            y_pred.append(preds)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_dataloader)
        epoch_acc = running_corrects.double() / len(test_dataloader)
        y = torch.cat(y_true)
        y_hat = torch.cat(y_pred)

        epoch_balanced_acc = metrics.balanced_accuracy_score(y.cpu(), y_hat.cpu())

        print(
            "Loss: {:.4f} Acc: {:.4f} Balanced acc: {:.4f}".format(
                epoch_loss, epoch_acc, epoch_balanced_acc
            )
        )
