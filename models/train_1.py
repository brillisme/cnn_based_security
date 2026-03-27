import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
from models.MIXPOOL import MIXPOOL


from models.UNSWNet7 import UNSWNet7Deep
# ---------------------------
# 网络结构：三层卷积 + 两次池化 + 全连接输出10类
# ---------------------------

# ---------------------------
# Helper Functions
# ---------------------------
def load_npz_data(path):
    """
    从 npz 文件加载数据，并转换为 torch 张量。
    要求 npz 文件中包含 "X" 和 "y" 两个数组。
    """
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor

def get_param_stats(model):
    """
    统计模型中每个模块的参数量。
    返回一个字典，键为模块名称，值为参数总数。
    """
    stats = {}
    for name, module in model.named_modules():
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if param_count > 0:
            stats[name] = param_count
    return stats

def get_new_output_dir(base_dir):
    """
    如果 base_dir 已存在，则自动生成 base_dir1、base_dir2 ... 直到找到不存在的目录名称。
    返回最终目录路径，并创建该目录。
    """
    output_dir = base_dir
    i = 0
    while os.path.exists(output_dir):
        i += 1
        output_dir = f"{base_dir}{i}"
    os.makedirs(output_dir)
    return output_dir

# 示例用法：
# new_output_dir = get_new_output_dir("run")
# 这样第一次训练会生成 "run" 目录，下一次会生成 "run1" 目录，以此类推。
'''
def compute_map(y_true, y_scores, num_classes):
    """
    计算多分类任务的 mAP (mean Average Precision)。
    y_true: numpy 数组 (n_samples,) 的真实类别索引。
    y_scores: numpy 数组 (n_samples, num_classes) 的预测概率（需经过 softmax）。
    num_classes: 类别总数。
    """
    y_true_onehot = np.zeros_like(y_scores)
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1
    APs = []
    for c in range(num_classes):
        ap = average_precision_score(y_true_onehot[:, c], y_scores[:, c])
        APs.append(ap)
    return np.mean(APs)
'''
# ---------------------------
# Main Training Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a CNN-based network security detection model.")
    # 数据集 YAML 文件
    parser.add_argument('--data_yaml', type=str, default="E:/cnn_based_network_security_detection_model/Data/processed_data/UNSW_NB15.yaml",
                        help='Path to dataset YAML config file')
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=13, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    # 恢复训练权重（可选）
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from (optional)')
    # 输出目录
    parser.add_argument('--output_dir', type=str, default='run', help='Directory to save outputs')
    args = parser.parse_args()

    # 创建输出目录
    args.output_dir = get_new_output_dir(args.output_dir)
    print(f"Output directory: {args.output_dir}")
    # ---------------------------
    # 读取数据集 YAML 配置
    # ---------------------------
    with open(args.data_yaml, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    dataset_path = data_cfg['path']         # 数据根目录
    train_file = data_cfg['train']          # 训练集
    val_file = data_cfg.get('val', None)    # 验证集（可选）
    test_file = data_cfg.get('test', None)  # 测试集（可选）
    num_classes = data_cfg.get('nc', 10)    # 类别数
    names = data_cfg.get('names', None)     # 映射：类别索引 -> 类别名称（可选）

    # ---------------------------
    # 加载数据
    # ---------------------------
    # 训练数据
    train_path = os.path.join(dataset_path, train_file)
    X_train, y_train = load_npz_data(train_path)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 验证数据
    if val_file:
        val_path = os.path.join(dataset_path, val_file)
        X_val, y_val = load_npz_data(val_path)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None

    # 测试数据
    if test_file:
        test_path = os.path.join(dataset_path, test_file)
        X_test, y_test = load_npz_data(test_path)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        test_loader = None

    # ---------------------------
    # 构建模型
    # ---------------------------
    model = UNSWNet7Deep(num_classes=num_classes)
    print(model)

    # ---------------------------
    # 设备设置
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------------------------
    # 加载预训练权重（如果有）
    # ---------------------------
    if args.resume is not None:
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)


    # ---------------------------
    # 定义损失函数、优化器
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # ---------------------------
    # 训练循环
    # ---------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * inputs.size(0)
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_epoch_loss:.4f}")
            # 保存验证集上表现最好的模型权重
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {epoch_loss:.4f}")

    # 保存最终模型权重
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))

    # ---------------------------
    # 评估测试集
    # ---------------------------
    if test_loader is not None:
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

        predictions = np.argmax(all_outputs, axis=1)
        acc = accuracy_score(all_targets, predictions)
        precision = precision_score(all_targets, predictions, average='macro')
        recall = recall_score(all_targets, predictions, average='macro')
        f1 = f1_score(all_targets, predictions, average='macro')
        # probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        # test_map = compute_map(all_targets, probs, num_classes)
        conf_mat = confusion_matrix(all_targets, predictions)

        total_samples = len(test_loader.dataset)
        inference_time = end_time - start_time
        speed = total_samples / inference_time

        print("===== Test Metrics =====")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        # print(f"Test mAP: {test_map:.4f}")
        print(f"Inference Speed: {speed:.2f} samples/second")
        print("Test Confusion Matrix:")
        print(conf_mat)

        # 如果 names 存在，可以把预测/混淆矩阵转成可读标签
        if names:
            print("\nCategory Names:")
            for i, n in names.items():
                print(f" {i}: {n}")

        with open(os.path.join(args.output_dir, "test_metrics.txt"), "w") as f:
            f.write(f"Test Accuracy: {acc:.4f}\n")
            f.write(f"Test Precision: {precision:.4f}\n")
            f.write(f"Test Recall: {recall:.4f}\n")
            f.write(f"Test F1 Score: {f1:.4f}\n")
            # f.write(f"Test mAP: {test_map:.4f}\n")
            f.write(f"Inference Speed: {speed:.2f} samples/second\n")
            f.write("Test Confusion Matrix:\n")
            f.write(np.array2string(conf_mat))

    # ---------------------------
    # 评估验证集（使用最佳模型权重）
    # ---------------------------
    if val_loader is not None:
        print("Evaluating on validation set using best model weights...")
        best_model = UNSWNet7Deep(num_classes=num_classes)
        best_model.to(device)
        best_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device))
        best_model.eval()

        all_outputs = []
        all_targets = []
        start_time = time.time()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = best_model(inputs)
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        end_time = time.time()
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        predictions = np.argmax(all_outputs, axis=1)
        acc_val = accuracy_score(all_targets, predictions)
        precision_val = precision_score(all_targets, predictions, average='macro')
        recall_val = recall_score(all_targets, predictions, average='macro')
        f1_val = f1_score(all_targets, predictions, average='macro')
        # probs_val = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        # val_map = compute_map(all_targets, probs_val, num_classes)
        conf_mat_val = confusion_matrix(all_targets, predictions)

        total_samples_val = len(val_loader.dataset)
        inference_time_val = end_time - start_time
        speed_val = total_samples_val / inference_time_val

        print("===== Validation Metrics =====")
        print(f"Validation Accuracy: {acc_val:.4f}")
        print(f"Validation Precision: {precision_val:.4f}")
        print(f"Validation Recall: {recall_val:.4f}")
        print(f"Validation F1 Score: {f1_val:.4f}")
        # print(f"Validation mAP: {val_map:.4f}")
        print(f"Validation Inference Speed: {speed_val:.2f} samples/second")
        print("Validation Confusion Matrix:")
        print(conf_mat_val)

        if names:
            print("\nCategory Names:")
            for i, n in names.items():
                print(f" {i}: {n}")

        with open(os.path.join(args.output_dir, "val_metrics.txt"), "w") as f:
            f.write(f"Validation Accuracy: {acc_val:.4f}\n")
            f.write(f"Validation Precision: {precision_val:.4f}\n")
            f.write(f"Validation Recall: {recall_val:.4f}\n")
            f.write(f"Validation F1 Score: {f1_val:.4f}\n")
            # f.write(f"Validation mAP: {val_map:.4f}\n")
            f.write(f"Validation Inference Speed: {speed_val:.2f} samples/second\n")
            f.write("Validation Confusion Matrix:\n")
            f.write(np.array2string(conf_mat_val))

    # ---------------------------
    # 保存模型各模块的参数统计信息
    # ---------------------------
    param_stats = get_param_stats(model)
    with open(os.path.join(args.output_dir, "param_stats.txt"), "w") as f:
        for module_name, count in param_stats.items():
            f.write(f"{module_name}: {count} parameters\n")

    # ---------------------------
    # 绘制并保存训练和验证损失曲线
    # ---------------------------
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    if val_loader is not None:
        plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    plt.close()

    print("Training complete.")
    print(f"Best model saved as '{os.path.join(args.output_dir, 'best_model.pt')}'")
    print(f"Final model saved as '{os.path.join(args.output_dir, 'final_model.pt')}'")
    print("All evaluation metrics, parameter stats, and loss curve saved in the output directory.")

if __name__ == "__main__":
    main()
