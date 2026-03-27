import torch
import torch.nn as nn


class FCClassifier(nn.Module):
    """
    全连接分类器模块：
    - 首先将输入展平为 (batch_size, -1)
    - 然后通过 Dropout 层进行正则化
    - 最后通过全连接层输出10个类别的 logits
    """

    def __init__(self, in_features, num_classes=10, dropout=0.5):
        super(FCClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 如果输入 x 还未展平，则先展平，假设第一维为 batch_size
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 示例用法：
'''
if __name__ == "__main__":
    # 假设输入张量形状为 (batch_size, channels, sequence_length)
    x = torch.randn(8, 64, 10)  # 例如：8个样本，64个通道，序列长度10
    # 这里 in_features = 64 * 10 = 640
    classifier = FCClassifier(in_features=640, num_classes=10, dropout=0.5)
    y = classifier(x)
    print(y.shape)  # 输出: (8, 10)
'''