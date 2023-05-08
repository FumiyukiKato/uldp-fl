import torch
from sklearn import metrics

from flamby.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedIsic2019,
)

device = "cpu"

train_dataset = FedIsic2019(train=True, pooled=True)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
test_dataset = FedIsic2019(train=False, pooled=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)

seed = 0
torch.manual_seed(seed)
model = Baseline()
model = model.to(device)
loss = BaselineLoss()

# weighted focal loss
weights = [0] * 8
for x in train_dataset:
    weights[int(x[1])] += 1
N = len(train_dataset)
class_weights = torch.FloatTensor([N / weights[i] for i in range(8)]).to(device)
lossfunc = BaselineLoss(alpha=class_weights)

optimize_final_layer_only = False
if optimize_final_layer_only:
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.base_model._fc.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.base_model._fc.parameters(), lr=LR)
else:
    optimizer = torch.optim.Adam(model.base_model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5
)

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
