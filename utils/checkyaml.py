import os
import yaml

yaml_file  = "E:/cnn_based_network_security_detection_model/Data/processed_data/UNSW_NB15.yaml"
with open(yaml_file, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if "names" in config:
    # 将所有键转换为字符串
    label_mapping = {str(k): v for k, v in config["names"].items()}
    for i in range(10):
        print(f"{i}: {label_mapping.get(str(i), '无')}")
else:
    print("未找到标签映射信息。")

