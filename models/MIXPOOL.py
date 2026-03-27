import torch
import torch.nn as nn


class MIXPOOL(nn.Module):
    """
    MISPOOL Module:
    - 同时进行 1D 最大池化和平均池化
    - 输出为两者的平均值

    参数：
    - kernel_size: 池化核大小
    - stride: 池化步幅（默认为 None，此时与 kernel_size 相同）
    - padding: 池化填充（默认为 0）
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MIXPOOL, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        self.avgpool = nn.AvgPool1d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # 分别进行最大池化和平均池化
        max_out = self.maxpool(x)
        avg_out = self.avgpool(x)
        # 取二者平均
        out = 0.6* max_out + 0.4 * avg_out
        return out


# 示例用法：
'''
if __name__ == "__main__":
    # 假设输入张量形状为 (batch_size, channels, sequence_length)
    x = torch.randn(8, 32, 50)  # 例如：8 个样本，32 个通道，序列长度 50
    mispool = MISPOOL(kernel_size=2, stride=2)
    y = mispool(x)
    print(y.shape)  # 输出形状： (8, 32, 25)
'''
