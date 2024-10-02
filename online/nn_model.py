import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def get_last_hidden_layer_params(net):
    return net.fc2.weight, net.fc2.bias

def get_output_first_layers(net, x):
    with torch.no_grad():
        out = net.fc1(x)
        out = net.relu1(out)
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


def train(net, x_train, y_train, num_epochs=10, lr=1):
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

def train(net, x_train, y_train, num_epochs=50, lr=0.01, batch_size=8):
    # normalize between -1 and 1
    # if x_train.std() != 0 and y_train.std() != 0:
    #     x_train = (x_train - x_train.mean()) / x_train.std()
    #     y_train = (y_train - y_train.mean()) / y_train.std()
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = CustomLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # Assuming 'prices' is still the first column in 'inputs'
            prices = inputs[:, 0]

            outputs = net(inputs)
            loss = criterion(outputs, targets, prices)
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
