import os
import numpy as np
import yaml

# 设置文件路径（根据你的实际情况修改）
processed_dir = "E:/cnn_based_network_security_detection_model/Data/processed_data"
test_file = os.path.join(processed_dir, "test_data.npz")
yaml_file = os.path.join(processed_dir, "UNSW_NB15.yaml")

# 加载 test.npz 文件
test_data = np.load(test_file)
y_test = test_data["y"]

# 查看测试集中所有唯一的标签
unique_labels = np.unique(y_test)
print("Test.npz 中唯一的标签：", unique_labels)

# 加载 YAML 文件中的标签映射信息
with open(yaml_file, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if "names" in config:
    # 如果 YAML 文件中键可能为数字或字符串，这里统一转换为字符串
    label_mapping = {str(k): v for k, v in config["names"].items()}
    print("\nTest.npz 中标签对应的映射：")
    for label in unique_labels:
        mapping = label_mapping.get(str(label), "无")
        print(f"数字标签 {label} 对应类别: {mapping}")
else:
    print("YAML 配置中没有找到标签映射信息。")


