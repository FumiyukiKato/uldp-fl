import torch
from torch import nn
import torch.nn.functional as F
from dataset import CREDITCARD, HEART_DISEASE, LIGHT_MNIST, MNIST, TCGA_BRCA, BODY

from mylogger import logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# return model from string name
def create_model(model_name: str, dataset_name: str, seed: int = None) -> nn.Module:
    """
    Create a model from a string name

    Inputs:
        model_name(str): string name of the model
        dataset_name(str): string name of the dataset
        seed(int): random seed

    Outputs:
        global_model(nn.Module): model
    """
    if seed is not None:
        torch.manual_seed(seed)

    if dataset_name == HEART_DISEASE:
        import flamby_utils.heart_disease as heart_disease

        model = heart_disease.custom_model()

    elif dataset_name == TCGA_BRCA:
        import flamby_utils.tcga_brca as tcga_brca

        model = tcga_brca.custom_model()

    elif dataset_name == CREDITCARD:
        model = PrivateFraudNet(30, 30, 4)

    elif dataset_name == BODY:
        input_dim = 11
        output_dim = 4
        model = BodyModel(input_dim, output_dim)

    elif model_name == "cnn":
        # Convolutional neural network
        if dataset_name == MNIST or dataset_name == LIGHT_MNIST:
            model = CNNMnist()
        else:
            raise NotImplementedError

    logger.debug(f"Number of model params (Dimension d): {count_parameters(model)}")
    return model


class PrivateFraudNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.GroupNorm(1, hidden_dim)
        )
        # make the number of hidden dim layers configurable
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.GroupNorm(1, hidden_dim))
            self.layers.append(nn.Dropout(0.5))

        # final layer
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out = self.input(x)
        for layer in self.layers:
            out = layer(out)
        return self.fc(out)


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class BodyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BodyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 35)
        self.fc3 = nn.Linear(35, 50)
        self.fc4 = nn.Linear(50, 65)
        self.dropout1 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(65, 80)
        self.dropout2 = nn.Dropout(0.2)
        self.fc6 = nn.Linear(80, 55)
        self.fc7 = nn.Linear(55, 35)
        self.fc8 = nn.Linear(35, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.dropout1(torch.relu(self.fc4(x)))
        x = self.dropout2(torch.relu(self.fc5(x)))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x
