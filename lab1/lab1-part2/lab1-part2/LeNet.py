import time
from abc import ABC

import torch
import torch.nn as nn
import torch.utils.data as data
import util


torch.manual_seed(1)
epochs = 30
batch_size = 5
learning_rate = 0.001
weight_decay = 0
super_argument = [epochs, batch_size, learning_rate, weight_decay]
train_data = util.get_data('train')
develop_data = util.get_data('develop')
test_data = util.get_data('test')


class LeNet(nn.Module, ABC):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.output = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=32 * 7 * 7, out_features=12),
            # nn.Linear(470, 329),
            # nn.Linear(329, 12),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.output(x)
        return output


    def save_network(self, path):
        torch.save(self.state_dict(), path)


    def load_network(self, path):
        self.load_state_dict(torch.load(path))


def train():
    print(cnn)
    # cnn.load_network('network.save')
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(cnn.parameters(), learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.LBFGS(cnn.parameters(), learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(cnn.parameters(), learning_rate, weight_decay=weight_decay)

    loss_function = nn.CrossEntropyLoss()

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    for epoch in range(epochs):
        print("epoch {}".format(epoch + 1))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = cnn(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test(develop_data)


def test(dataset):
    data_loader = data.DataLoader(dataset=dataset)
    correct_num = 0
    for step, (test_x, test_y) in enumerate(data_loader):
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        test_output = cnn(test_x)
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
        if pred_y == test_y.item():
            correct_num += 1
    print("accuracy: ", correct_num / len(dataset))


if __name__ == '__main__':
    print(super_argument)
    print(torch.cuda.is_available())
    cnn = LeNet()
    cnn.cuda()
    start = time.time()
    train()
    end = time.time()
    print("TIME:", (end - start))
    print("TEST PERFORMANCE:")
    test(test_data)
