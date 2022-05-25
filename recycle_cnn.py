import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.act = torch.nn.ReLU()

        # our image size:
        # 3 * 384 * 512
        # rgb * H * W

        # 32 channels of 192*256
        self.conv1 = torch.nn.Conv2d(3, 32, 7, 1, 3)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        # 32 channels of 96*128
        self.conv2 = torch.nn.Conv2d(32, 32, 11, 1, 5)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        # 32 channels of 48*64
        self.conv3 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.pool3 = torch.nn.MaxPool2d(2, 2)

        # 32 channels of 24*32
        self.conv4 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.pool4 = torch.nn.MaxPool2d(2, 2)

        # 32*24*32 => 6 classes
        self.fc1 = torch.nn.Linear(32 * 24 * 32, 128)
        self.drop = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 6)

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
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act(out)
        out = self.pool4(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out
