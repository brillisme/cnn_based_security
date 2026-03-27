import torch
import torch.nn as nn
from models.MIXPOOL import MIXPOOL




class UNSWNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UNSWNet, self).__init__()
        # 输入: (batch, 49) -> (batch, 32, 207)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = MIXPOOL(kernel_size=2, stride=2)  # 长度减半

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = MIXPOOL(kernel_size=2, stride=2)  # 再减半

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.skip = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(128)
        )

        # 最后特征长度: 207 -> 103 -> 51，通道数 128 -> flatten 后 128*51
        self.fc = nn.Linear(128 * 48, num_classes)

    def forward(self, x):
        # print("input:", x.shape)
        # x: (batch, 49) -> (batch, 1, 49)
        out1 = self.conv1(x.unsqueeze(1))  # (batch, 32, 49)
        # print("after conv1:", x.shape)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)  # (batch, 32, 24)
        # print("after pool1:", x.shape)  # (batch, 32, 24) if stride=2

        out2 = self.conv2(out1)  # (batch, 64, 24)
        # print("after conv2:", x.shape)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        out2 = self.pool2(out2)  # (batch, 64, 12)
        # print("after pool2:", x.shape)

        out3 = self.conv3(out2)  # (batch, 128, 12)
        # print("after conv3:", x.shape)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)

        skip_out = self.skip(out1)
        if skip_out.size(2) > out3.size(2):
            skip_out = skip_out[:, :, :out3.size(2)]
        out = out3 + skip_out  # (batch, 128, 12)
        out = self.relu(out)  # 再做一次 ReLU

        out = torch.flatten(out, 1)  # (batch, 128*12)
        # print("after flatten:", x.shape)
        out = self.fc(out)           # (batch, num_classes)
        # print("after fc:", x.shape)
        return out