import torch
import numpy as np
from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    Optimizer,
)
from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset

# Load dataset
train_dataset = FedDataset(center=0, train=True, pooled=True)
test_dataset = FedDataset(center=0, train=False, pooled=True)

# Create model
lossfunc = BaselineLoss()
model = Baseline()
optimizer = Optimizer(model.parameters(), lr=LR)

if __name__ == "__main__":
    # Train
    device = "cpu"
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    for epoch in range(0, NUM_EPOCHS_POOLED):
        for idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = lossfunc(outputs, y)
            loss.backward()
            optimizer.step()

    # Test
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
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
