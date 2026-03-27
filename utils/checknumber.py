import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置预处理后数据的存放路径
processed_dir = "E:/cnn_based_network_security_detection_model/Data/processed_data"

# 加载预处理后的训练集和测试集 npz 文件
train_file = os.path.join(processed_dir, "train_data.npz")
test_file  = os.path.join(processed_dir, "test_data.npz")

train_data = np.load(train_file)
test_data = np.load(test_file)

# 提取特征和标签
X_train = train_data["X"]
y_train = train_data["y"]
X_test  = test_data["X"]
y_test  = test_data["y"]

# 打印数据形状
print("训练集特征形状:", X_train.shape)
print("训练集标签形状:", y_train.shape)
print("测试集特征形状:", X_test.shape)
print("测试集标签形状:", y_test.shape)

# 将标签转换为 pandas 的 Series 方便统计
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

print("\n训练集标签分布:")
print(y_train_series.value_counts().sort_index())

print("\n测试集标签分布:")
print(y_test_series.value_counts().sort_index())

# 绘制测试集标签的直方图
plt.figure(figsize=(8, 4))
y_test_series.value_counts().sort_index().plot(kind='bar')
plt.xlabel("类别编码")
plt.ylabel("样本数量")
plt.title("测试集标签分布")
plt.show()

