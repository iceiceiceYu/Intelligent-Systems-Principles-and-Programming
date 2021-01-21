from abc import ABC

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import util


# torch.manual_seed(1)
epochs = 60
batch_size = 1024
learning_rate = 0.001
weight_decay = 0
super_argument = [epochs, batch_size, learning_rate, weight_decay]
train_data = util.get_data('train')
develop_data = util.get_data('develop')
test_data = util.get_data('test')
total_accuracy = []


class LeNet(nn.Module, ABC):
    def __init__(self):
        torch.manual_seed(1)
        super(LeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.output = nn.Sequential(
            nn.Linear(in_features=32 * 7 * 7, out_features=12),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.output(x)
        return output


class LeNetWithDropout(nn.Module, ABC):
    def __init__(self):
        torch.manual_seed(1)
        super(LeNetWithDropout, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.output = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=32 * 7 * 7, out_features=12),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.output(x)
        return output


class LeNetWithBatchNormalization(nn.Module, ABC):
    def __init__(self):
        torch.manual_seed(1)
        super(LeNetWithBatchNormalization, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.output = nn.Sequential(
            nn.Linear(in_features=32 * 7 * 7, out_features=12),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.output(x)
        return output


class AlexNet(nn.Module, ABC):
    def __init__(self):
        torch.manual_seed(1)
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.LocalResponseNorm(16),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.LocalResponseNorm(32),
            nn.Conv2d(32, 96, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 12),
            # nn.Dropout(),
            # nn.Linear(2048, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, 12),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


def train(network):
    print(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_total_accuracy = []
    for epoch in range(epochs):
        print("epoch {}".format(epoch + 1))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = network(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = test(network, test_data)
        # accuracy = test(network, develop_data)
        dev_total_accuracy.append(accuracy)
        accuracy = test(network, test_data)
        total_accuracy.append(accuracy)
    return dev_total_accuracy


def train_with_L1Reg(network):
    print(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    # regularization_loss = 0
    # for param in network.parameters():
    # regularization_loss += torch.sum(abs(param))
    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_total_accuracy = []
    for epoch in range(epochs):
        print("epoch {}".format(epoch + 1))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            regularization_loss = 0
            for param in network.parameters():
                regularization_loss += torch.sum(abs(param))

            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = network(batch_x)

            classify_loss = loss_function(output, batch_y)
            loss = classify_loss + 0.001 * regularization_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = test(network, develop_data)
        dev_total_accuracy.append(accuracy)
        accuracy = test(network, test_data)
        total_accuracy.append(accuracy)
    return dev_total_accuracy


def train_with_L2Reg(network):
    print(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_function = nn.CrossEntropyLoss()

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_total_accuracy = []
    for epoch in range(epochs):
        print("epoch {}".format(epoch + 1))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = network(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = test(network, develop_data)
        dev_total_accuracy.append(accuracy)
        accuracy = test(network, test_data)
        total_accuracy.append(accuracy)
    return dev_total_accuracy


def test(network, dataset):
    data_loader = data.DataLoader(dataset=dataset)
    correct_num = 0
    for step, (test_x, test_y) in enumerate(data_loader):
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        test_output = network(test_x)
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
        if pred_y == test_y.item():
            correct_num += 1
    accuracy = correct_num / len(dataset)
    print("accuracy: ", accuracy)
    return accuracy


if __name__ == '__main__':
    print(super_argument)
    print(torch.cuda.is_available())
    cnn1 = LeNet()
    cnn1.cuda()
    # cnn2 = AlexNet()
    # cnn2.cuda()
    # cnn2 = LeNetWithDropout()
    # cnn2.cuda()
    cnn2 = LeNetWithBatchNormalization()
    cnn2.cuda()
    total_accuracy_cnn1 = train(cnn1)
    total_accuracy_cnn2 = train(cnn2)

    print("TEST PERFORMANCE:")
    test(cnn1, test_data)
    test(cnn2, test_data)

    plt.plot(np.arange(1, 1 + epochs).astype(dtype=np.str), total_accuracy_cnn1, color='lightblue',
             label='cnn without batch normalization')
    plt.plot(np.arange(1, 1 + epochs).astype(dtype=np.str), total_accuracy_cnn2, color='violet',
             label='cnn with batch normalization')
    # plt.plot(np.arange(1, 1 + epochs).astype(dtype=np.str), total_accuracy_cnn2, color='lightblue',
    #          label='develop set accuracy')
    # plt.plot(np.arange(1, 1 + epochs).astype(dtype=np.str), total_accuracy, color='violet',
    #          label='test set accuracy')
    plt.xlabel('epoch')
    # plt.ylabel('network train with L2 regularization')
    plt.ylabel('test set accuracy')
    plt.legend()
    plt.show()
