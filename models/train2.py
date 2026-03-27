import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    average_precision_score
from models.MIXPOOL import MIXPOOL
from models.UNSWNet7 import UNSWNet7Deep


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


# ---------------------------
# 绘图函数（验证指标和混淆矩阵）
# ---------------------------
def plot_val_metrics(epochs, val_acc, val_precision, val_recall, val_f1, save_dir=None):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_acc, marker='o', label='Val Accuracy')
    plt.plot(epochs, val_precision, marker='o', label='Val Precision')
    plt.plot(epochs, val_recall, marker='o', label='Val Recall')
    plt.plot(epochs, val_f1, marker='o', label='Val F1 Score')
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Validation Metrics over Epochs")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        save_path = os.path.join(save_dir, "val_metrics.png")
        plt.savefig(save_path)
        print("验证指标图已保存到:", save_path)
    plt.show()


def plot_confusion_matrices(confusion_matrices, save_dir=None):
    num_epochs = len(confusion_matrices)
    cols = 4  # 每行显示 4 个混淆矩阵
    rows = math.ceil(num_epochs / cols)
    plt.figure(figsize=(cols * 4, rows * 4))
    for i, cm in enumerate(confusion_matrices, 1):
        plt.subplot(rows, cols, i)
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
        plt.title(f"Epoch {i}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
    plt.tight_layout()
    if save_dir is not None:
        save_path = os.path.join(save_dir, "val_confusion_matrices.png")
        plt.savefig(save_path)
        print("验证集混淆矩阵图已保存到:", save_path)
    plt.show()


# ---------------------------
# Main Training Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a CNN-based network security detection model.")
    # 数据集 YAML 文件
    parser.add_argument('--data_yaml', type=str,
                        default="E:/cnn_based_network_security_detection_model/Data/processed_data/UNSW_NB15.yaml",
                        help='Path to dataset YAML config file')
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # 恢复训练权重（可选）
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (optional)')
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

    dataset_path = data_cfg['path']  # 数据根目录
    train_file = data_cfg['train']  # 训练集
    val_file = data_cfg.get('val', None)  # 验证集
    test_file = data_cfg.get('test', None)  # 测试集
    num_classes = data_cfg.get('nc', 10)  # 类别数
    names = data_cfg.get('names', None)  # 映射：类别索引 -> 类别名称

    # ---------------------------
    # 加载数据
    # ---------------------------
    # 训练数据
    train_path_full = os.path.join(dataset_path, os.path.basename(train_file))
    X_train, y_train = load_npz_data(train_path_full)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 验证数据
    if val_file:
        val_path_full = os.path.join(dataset_path, os.path.basename(val_file))
        X_val, y_val = load_npz_data(val_path_full)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None

    # 测试数据
    if test_file:
        test_path_full = os.path.join(dataset_path, os.path.basename(test_file))
        X_test, y_test = load_npz_data(test_path_full)
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

    # 记录每个 epoch 的验证集分类指标与混淆矩阵
    val_acc_list = []
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []
    val_conf_mat_list = []

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
            all_outputs = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * inputs.size(0)
                    all_outputs.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_epoch_loss)
            # 计算验证集分类指标
            all_outputs = np.concatenate(all_outputs, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            predictions = np.argmax(all_outputs, axis=1)
            acc_val = accuracy_score(all_targets, predictions)
            precision_val = precision_score(all_targets, predictions, average='macro')
            recall_val = recall_score(all_targets, predictions, average='macro')
            f1_val = f1_score(all_targets, predictions, average='macro')
            conf_mat_val = confusion_matrix(all_targets, predictions)
            val_acc_list.append(acc_val)
            val_precision_list.append(precision_val)
            val_recall_list.append(recall_val)
            val_f1_list.append(f1_val)
            val_conf_mat_list.append(conf_mat_val)
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_epoch_loss:.4f}")
            # 保存验证集上表现最好的模型权重
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {epoch_loss:.4f}")

    # 保存最终模型权重
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))

    # ---------------------------
    # 绘制并保存训练和验证 Loss 曲线
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

    # ---------------------------
    # 绘制并保存验证集分类指标图（Accuracy, Precision, Recall, F1）
    # ---------------------------
    epochs_list = list(range(1, epochs + 1))
    plot_val_metrics(epochs_list, val_acc_list, val_precision_list, val_recall_list, val_f1_list,
                     save_dir=args.output_dir)

    # 绘制并保存每个 epoch 的验证集混淆矩阵图
    plot_confusion_matrices(val_conf_mat_list, save_dir=args.output_dir)

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
                all_targets.append(targets.cpu().numpy())
        end_time = time.time()
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        predictions = np.argmax(all_outputs, axis=1)
        acc = accuracy_score(all_targets, predictions)
        precision = precision_score(all_targets, predictions, average='macro')
        recall = recall_score(all_targets, predictions, average='macro')
        f1 = f1_score(all_targets, predictions, average='macro')
        conf_mat = confusion_matrix(all_targets, predictions)

        total_samples = len(test_loader.dataset)
        inference_time = end_time - start_time
        speed = total_samples / inference_time

        print("===== Test Metrics =====")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Inference Speed: {speed:.2f} samples/second")
        print("Test Confusion Matrix:")
        print(conf_mat)

        if names:
            print("\nCategory Names:")
            for i, n in names.items():
                print(f" {i}: {n}")

        with open(os.path.join(args.output_dir, "test_metrics.txt"), "w") as f:
            f.write(f"Test Accuracy: {acc:.4f}\n")
            f.write(f"Test Precision: {precision:.4f}\n")
            f.write(f"Test Recall: {recall:.4f}\n")
            f.write(f"Test F1 Score: {f1:.4f}\n")
            f.write(f"Inference Speed: {speed:.2f} samples/second\n")
            f.write("Test Confusion Matrix:\n")
            f.write(np.array2string(conf_mat))

    # ---------------------------
    # 保存模型各模块的参数统计信息
    # ---------------------------
    param_stats = get_param_stats(model)
    with open(os.path.join(args.output_dir, "param_stats.txt"), "w") as f:
        for module_name, count in param_stats.items():
            f.write(f"{module_name}: {count} parameters\n")

    print("Training complete.")
    print(f"Best model saved as '{os.path.join(args.output_dir, 'best_model.pt')}'")
    print(f"Final model saved as '{os.path.join(args.output_dir, 'final_model.pt')}'")
    print("All evaluation metrics, parameter stats, loss curve, and validation plots saved in the output directory.")


if __name__ == "__main__":
    main()
