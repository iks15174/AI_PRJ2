import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.act = torch.nn.ReLU()

        # our image size:
        # 3 * 227 * 227
        # rgb * H * W

        # 96 channels of 27*27
        self.conv1 = torch.nn.Conv2d(3, 96, 11, 4, 0)  # 227->55
        self.bn1 = torch.nn.BatchNorm2d(96)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)  # 55->27

        # 256 channels of 13*13
        self.conv2 = torch.nn.Conv2d(96, 256, 5, 1, 2)  # 27->27
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)  # 27->13

        # 256 channels of 6*6
        self.conv3 = torch.nn.Conv2d(256, 384, 3, 1, 1)
        self.bn3 = torch.nn.BatchNorm2d(384)
        self.conv4 = torch.nn.Conv2d(384, 384, 3, 1, 1)
        self.bn4 = torch.nn.BatchNorm2d(384)
        self.conv5 = torch.nn.Conv2d(384, 256, 3, 1, 1)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)  # 13->6

        # 256*6*6 => 6 classes
        self.fc1 = torch.nn.Linear(256 * 6 * 6, 4096)
        self.drop = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, 6)

        # torch.nn.init.kaiming_uniform_(self.fc1.weight)
        # torch.nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.act(out)
        out = self.pool3(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc3(out)

        return out
