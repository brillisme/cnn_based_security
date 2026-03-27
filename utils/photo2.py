import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义混淆矩阵数据
conf_mat = np.array([
    [7666, 1710, 1743,  182,    2,    0,    6,   35,    0,    0],
    [ 956, 8653, 1313,  146,    6,    0,    0,   53,    2,    1],
    [1405, 1727, 6227, 1436,   46,   17,    3,  126,   45,    5],
    [ 490,  663, 2838, 6744,   98,   32,   19,  249,   80,   50],
    [ 192,  216,  388,  242, 9805,   11,  233,   80,  132,    9],
    [   8,   21,   69,   59,   11,11001,    0,    5,    3,    2],
    [  28,    1,   12,   18,  213,    0,10884,   11,    9,    0],
    [ 316,  723,  760,  261,   27,    2,    2,9250,   16,    1],
    [   0,    0,    0,    5,    8,    0,    1, 105,11015,    3],
    [   0,    0,    0,    0,    0,    0,    0,   0,    0,11068]
])

# 绘制混淆矩阵热图
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
