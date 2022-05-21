import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # our image size:
        # 3 * 384 * 512
        # rgb * H * W

        # 32 channels of 192*256
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        # 64 channels of 96*128
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        # 64*96*128 => 6 classes
        self.fc1 = torch.nn.Linear(64*96*128, 128)
        self.drop = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(128, 6)

        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out
