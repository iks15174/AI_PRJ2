import torch
from torch import nn
import torchvision.models as models


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # n*n*in_channels의 input을 n*n*out_channels로 변환
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # n*n*out_channels의 input을 n*n*out_channels로 변환
            # BasicBlock.expasion(1) 변수는 현재 코드에선 역할을 하지 않음
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # shortcut block(더해지는 block)을 정의한다.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # CNN(residual_function)을 거친 layer의 크기가 shortcut block과 크기가 다를 경우
        # shortcut의 block을 CNN(residual_function)을 거친 layer의 크기와 같게 만들어 준다.
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=6):
        super().__init__()

        # our image size:
        # 3 * 384 * 512
        # rgb * H * W

        self.in_channels = 64

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer2 = self._make_layer(block, 64, num_block[0], 1)
        self.layer3 = self._make_layer(block, 128, num_block[1], 2)
        self.layer4 = self._make_layer(block, 256, num_block[2], 2)
        self.layer5 = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        x = self.layer3(output)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


PretrainedAlexnet = models.alexnet(pretrained=True)
PretrainedAlexnet.classifier[-1] = torch.nn.Linear(4096, 6)

PretrainedAlexnet_binary = models.alexnet(pretrained=True)
PretrainedAlexnet_binary.classifier[-1] = torch.nn.Linear(4096, 2)

PretrainedResnet = models.resnet18(pretrained=True)
num_ftrs = PretrainedResnet.fc.in_features
PretrainedResnet.fc = torch.nn.Linear(num_ftrs, 6)

PretrainedResnet_binary = models.resnet18(pretrained=True)
num_ftrs = PretrainedResnet_binary.fc.in_features
PretrainedResnet_binary.fc = torch.nn.Linear(num_ftrs, 2)
