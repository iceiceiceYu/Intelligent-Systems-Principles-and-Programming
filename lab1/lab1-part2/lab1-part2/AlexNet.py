import time
from abc import ABC

import torch
import torch.nn as nn
import torch.utils.data as data
import util


torch.manual_seed(1)
epochs = 100
batch_size = 128
learning_rate = 0.001
weight_decay = 0
super_argument = [epochs, batch_size, learning_rate, weight_decay]
train_data = util.get_data('train')
develop_data = util.get_data('develop')
test_data = util.get_data('test')


class AlexNet(nn.Module, ABC):
    def __init__(self):
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


    def save_network(self, path):
        torch.save(self.state_dict(), path)


    def load_network(self, path):
        self.load_state_dict(torch.load(path))


def train():
    print(cnn)
    # cnn.load_network('network.save')
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    for epoch in range(epochs):
        total_loss = 0
        print("epoch {}".format(epoch + 1))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = cnn(batch_x)
            loss = loss_function(output, batch_y)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("total loss: {}".format(loss))
        print("average loss: {}".format(loss / len(train_data)))
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
    cnn = AlexNet()
    cnn.cuda()
    start = time.time()
    train()
    end = time.time()
    print("TIME:", (end - start))
    print("TEST PERFORMANCE:")
    test(test_data)
