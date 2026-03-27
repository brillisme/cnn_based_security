import torch
import torch.nn as nn
from models.MIXPOOL import MIXPOOL  # 请确保 MIXPOOL 实现正确


class UNSWNet7Deep(nn.Module):
    def __init__(self, num_classes=10):
        super(UNSWNet7Deep, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 使用 MIXPOOL 做池化，假设行为类似于 MaxPool1d(kernel_size=2, stride=2)
        self.pool = MIXPOOL(kernel_size=2, stride=2)

        # --- 主分支 ---
        # 第1层卷积：输入 1 -> 32，保持宽度 192
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        # 池化后宽度从 192 变为 96

        # 第2层卷积：32 -> 64，宽度保持96
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # 第3层卷积：64 -> 128，宽度保持96；池化后变为48
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        # 池化后：宽度 48

        # --- 主分支继续 ---
        # 第4层卷积：128 -> 256，宽度保持48
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        # 第5层卷积：256 -> 256，宽度保持48
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        # 池化后：宽度从48变为24
        # 注意这里使用相同的池化层
        # 这里我们对 conv5 输出进行池化
        # 主分支后续操作均在池化后的结果上进行
        # 第6层卷积：256 -> 256，宽度保持24
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(256)

        # 第7层卷积：256 -> 256，宽度保持24
        self.conv7 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm1d(256)

        # --- 跳跃分支 ---
        # 从第3层卷积池化后的输出（out3_pool，形状：(batch,128,48)）建立跳跃分支
        # 使用 1×1 卷积和 stride=2 将其转换为 (batch,256,24)
        self.skip = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(256)
        )

        # --- 全连接层 ---
        # 主分支最终输出为 (batch,256,24)，flatten 后维度 256*24=6144
        self.fc = nn.Linear(256 * 24, num_classes)

    def forward(self, x):
        # 输入 x: (batch, 192)
        x = x.unsqueeze(1)  # 变为 (batch,1,192)

        # 主分支第1层
        out1 = self.conv1(x)  # (batch,32,192)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1_pool = self.pool(out1)  # (batch,32,96)

        # 第2层
        out2 = self.conv2(out1_pool)  # (batch,64,96)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        # 第3层
        out3 = self.conv3(out2)  # (batch,128,96)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)
        out3_pool = self.pool(out3)  # (batch,128,48)

        # 保存跳跃分支
        skip_out = self.skip(out3_pool)  # (batch,256,24)

        # 第4层
        out4 = self.conv4(out3_pool)  # (batch,256,48)
        out4 = self.bn4(out4)
        out4 = self.relu(out4)

        # 第5层
        out5 = self.conv5(out4)  # (batch,256,48)
        out5 = self.bn5(out5)
        out5 = self.relu(out5)
        out5_pool = self.pool(out5)  # (batch,256,24)

        # 第6层
        out6 = self.conv6(out5_pool)  # (batch,256,24)
        out6 = self.bn6(out6)
        out6 = self.relu(out6)

        # 第7层
        out7 = self.conv7(out6)  # (batch,256,24)
        out7 = self.bn7(out7)
        out7 = self.relu(out7)

        # 将主分支输出与跳跃分支相加
        out = out7 + skip_out
        out = self.relu(out)

        # flatten 后送入全连接层
        out = out.flatten(1)  # (batch,256*24)
        out = self.fc(out)
        return out


# 测试代码
'''
if __name__ == '__main__':
    model = UNSWNet7Deep(num_classes=10)
    test_input = torch.randn(64, 192)  # 假设 batch_size=64，输入尺寸 192
    test_output = model(test_input)
    print("输出形状：", test_output.shape)  # 预期 (64, 10)
'''


