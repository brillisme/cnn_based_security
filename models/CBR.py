import torch
import torch.nn as nn


class CBR(nn.Module):
    """
    CBR Module:
    - C: 1D Convolution layer (without bias, as BN 后通常无需 bias)
    - B: Batch Normalization (1D)
    - R: ReLU activation (inplace)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CBR, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 示例用法：
'''
if __name__ == "__main__":
    # 假设输入张量形状为 (batch_size, in_channels, sequence_length)
    x = torch.randn(8, 16, 50)  # 例如：8个样本，16个通道，序列长度50
    cbr = CBR(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    y = cbr(x)
    print(y.shape)  # 预期输出: (8, 32, 50)
'''
