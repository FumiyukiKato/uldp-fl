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


device = "cpu"

training_dataloader = DataLoader(
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
model = Baseline()
model = model.to(device)
loss = BaselineLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


for e in tqdm(range(NUM_EPOCHS_POOLED)):
    for s, (X, y) in enumerate(training_dataloader):
        # traditional training loop with optional GPU transfer
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        lm = loss(y_pred, y)
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
