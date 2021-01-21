import time
from abc import ABC

import torch
import torch.nn as nn
import torch.utils.data as data

import util


torch.manual_seed(1)
epochs = 20
batch_size = 5
learning_rate = 0.001
weight_decay = 0
super_argument = [epochs, batch_size, learning_rate, weight_decay]
train_data = util.get_data('total_train')
test_data = util.get_data('interview_test')
pred_file = open('pred.txt', 'w')

map = {0: 1, 1: 10, 2: 11, 3: 12, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}


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


    def save_network(self, path):
        torch.save(self.state_dict(), path)


    def load_network(self, path):
        self.load_state_dict(torch.load(path))


def train():
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    for epoch in range(epochs):
        print("epoch {}".format(epoch + 1))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    cnn.save_network('interview3.nn')


def test(dataset):
    data_loader = data.DataLoader(dataset=dataset)
    for step, (test_x, test_y) in enumerate(data_loader):
        test_output = cnn(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        pred_file.write(str(map[int(pred_y)]) + '\n')


if __name__ == '__main__':
    print(super_argument)
    print(torch.cuda.is_available())
    cnn = LeNet()
    cnn.load_network('interview3.nn')
    # start = time.time()
    # train()
    # end = time.time()
    # print("TIME:", (end - start))
    test(test_data)
