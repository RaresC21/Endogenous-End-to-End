import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 10)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


def get_last_hidden_layer_params(net):
    return net.fc4.weight, net.fc4.bias

def get_output_first_layers(net, x):
    with torch.no_grad():
        out = net.fc1(x)
        out = net.relu1(out)
        out = net.fc2(out)
        out = net.relu2(out)
        out = net.fc3(out)
        out = net.relu3(out)
    return out


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, price):
        output = output.squeeze()
        g_output = output * price
        g_target = target * price
        mse_loss = nn.functional.mse_loss(g_output, g_target)
        return mse_loss


def train(net, x_train, y_train, num_epochs=100, lr=0.01):
    x_train = (x_train - x_train.mean()) / x_train.std()
    y_train = (y_train - y_train.mean()) / y_train.std()

    criterion = CustomLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = net(x_train)
        prices = x_train[:, 0]   # if more than one feature, price is the first column
        loss = criterion(outputs, y_train, prices)
        loss.backward()
        optimizer.step()

    return net

if __name__ == '__main__':
    ##### TEST #####
    
    # synthetic data
    def generate_data(num_samples=1000, num_features=1, noise=0.1):
        x = torch.randn(num_samples, num_features)
        y = torch.randn(num_samples, num_features)
        return x, y

    x, y = generate_data()
    net = Net()
    net = train(net, x, y)
    print(net)
    print(net.fc3.weight)
    print(net.fc3.bias)
