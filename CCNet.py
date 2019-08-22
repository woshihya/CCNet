import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, avg_size=5):
        super().__init__()
        self.avg_size = avg_size

        self.function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            nn.ReLU(inplace=True)
        )

        self.scaler = nn.Sequential(
            nn.Linear(out_channels * self.avg_size ** 2, out_channels // 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels // 16),
            nn.Linear(out_channels // 16, out_channels * 2),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d(self.avg_size)

        self.adjust = nn.Sequential()

        if stride != 1 and in_channels != out_channels:
            self.adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=1, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        elif stride != 1:
            self.adjust = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        elif in_channels != out_channels:
            self.adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        f = self.function(x)
        x = self.adjust(x)
        information = self.avg(x + f)
        information = information.view(information.size(0), -1)
        scaler = self.scaler(information) * 2
        scaler = scaler.reshape(scaler.size(0), scaler.size(1), 1, 1)
        scaler_f = scaler[:, :f.size(1), :, :]
        scaler_x = scaler[:, f.size(1):, :, :]
        return f * scaler_f + x * scaler_x


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, avg_size=5):
        super().__init__()
        self.avg_size = avg_size

        self.function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            nn.ReLU(inplace=True)
        )

        self.scaler = nn.Sequential(
            nn.Linear(out_channels * self.avg_size ** 2 * BottleNeck.expansion, out_channels // 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels // 16),
            nn.Linear(out_channels // 16, out_channels * 2 * BottleNeck.expansion),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d(self.avg_size)

        self.adjust = nn.Sequential()

        if stride != 1 and in_channels != out_channels * BottleNeck.expansion:
            self.adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=1, kernel_size=1, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=stride),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )
        elif stride != 1:
            self.adjust = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        elif in_channels != out_channels * BottleNeck.expansion:
            self.adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=1, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

    def forward(self, x):
        f = self.function(x)
        x = self.adjust(x)
        information = self.avg(x + f)
        information = information.view(information.size(0), -1)
        scaler = self.scaler(information) * 2
        scaler = scaler.reshape(scaler.size(0), scaler.size(1), 1, 1)
        scaler_f = scaler[:, :f.size(1), :, :]
        scaler_x = scaler[:, f.size(1):, :, :]
        return f * scaler_f + x * scaler_x


class CCNet(nn.Module):

    def __init__(self, block, num_block, init_channels, avg_size=5, num_classes=100):
        super().__init__()

        self.in_channels = init_channels
        self.avg_size = avg_size

        self.init_convolution = nn.Sequential(
            nn.Conv2d(3, init_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True))
        self.block1 = self._make_layer(block, init_channels, num_block[0], 1)
        self.block2 = self._make_layer(block, init_channels * 2, num_block[1], 2)
        self.block3 = self._make_layer(block, init_channels * 4, num_block[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(init_channels * 4 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.avg_size))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.init_convolution(x)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def CCNet16():
    return CCNet(BasicBlock, [2, 3, 2], 128)

def CCNet54():
    return CCNet(BasicBlock, [3, 20, 3], 160)

def CCNet149():
    return CCNet(BottleNeck, [3, 40, 6], 64, 3)