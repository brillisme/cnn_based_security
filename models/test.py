import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import importlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    average_precision_score


def load_npz_data(path):
    """
    从 npz 文件加载数据，并转换为 torch 张量。
    """
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def compute_map(y_true, y_scores, num_classes):
    """
    计算多分类任务的 mAP（mean Average Precision）。
    """
    y_true_onehot = np.zeros_like(y_scores)
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1
    APs = []
    for c in range(num_classes):
        ap = average_precision_score(y_true_onehot[:, c], y_scores[:, c])
        APs.append(ap)
    return np.mean(APs)


def main():
    parser = argparse.ArgumentParser(description="Test a trained CNN-based network security detection model.")
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to dataset YAML config file')
    parser.add_argument('--model_yaml', type=str, required=True, help='Path to model YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (e.g., best_model.pt)')
    args = parser.parse_args()

    # 加载 YAML 配置文件
    with open(args.data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model_yaml, 'r') as f:
        model_cfg = yaml.safe_load(f)

    dataset_path = data_cfg['path']
    test_file = data_cfg.get('test', None)
    num_classes = data_cfg.get('nc', 2)
    batch_size = data_cfg.get('batch_size', 32)

    if not test_file:
        raise ValueError("Test file not specified in data YAML.")

    # 加载测试数据
    test_path = f"{dataset_path}/{test_file}"
    X_test, y_test = load_npz_data(test_path)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 构建模型（与训练时保持一致）
    model_module_path = model_cfg.get('model_module')
    model_class_name = model_cfg.get('model_class')
    model_params = model_cfg.get('params', {})
    if not model_module_path or not model_class_name:
        raise ValueError("Model YAML must specify 'model_module' and 'model_class'.")

    model_module = importlib.import_module(model_module_path)
    ModelClass = getattr(model_module, model_class_name)
    model = ModelClass(**model_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载模型权重
    print(f"Loading model weights from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)

    # 测试评估
    model.eval()
    all_outputs = []
    all_targets = []
    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    end_time = time.time()
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算预测结果
    predictions = np.argmax(all_outputs, axis=1)

    # Accuracy, Precision, Recall, F1 Score
    acc = accuracy_score(all_targets, predictions)
    precision = precision_score(all_targets, predictions, average='macro')
    recall = recall_score(all_targets, predictions, average='macro')
    f1 = f1_score(all_targets, predictions, average='macro')

    # 使用 softmax 得到概率用于 mAP
    probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
    test_map = compute_map(all_targets, probs, num_classes)

    # 混淆矩阵
    conf_mat = confusion_matrix(all_targets, predictions)

    total_samples = len(test_dataset)
    inference_time = end_time - start_time
    speed = total_samples / inference_time

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test mAP: {test_map:.4f}")
    print(f"Inference Speed: {speed:.2f} samples/second")
    print("Confusion Matrix:")
    print(conf_mat)

    # 保存测试指标到文件
    with open("test_metrics.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
        f.write(f"Test mAP: {test_map:.4f}\n")
        f.write(f"Inference Speed: {speed:.2f} samples/second\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_mat))


if __name__ == "__main__":
    main()
