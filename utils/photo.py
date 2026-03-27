import matplotlib.pyplot as plt
import pandas as pd

# 构造数据
data = {
    "指标": ["Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Inference Speed"],
    "数值": ["0.8242", "0.8364", "0.8245", "0.8278", "34303.24 samples/second"]
}

df = pd.DataFrame(data)

# 创建图像并显示表格
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# 保存为图片
plt.savefig("metrics_table.png", dpi=300, bbox_inches="tight")
plt.show()
