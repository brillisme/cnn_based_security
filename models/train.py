import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import importlib
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    average_precision_score


# ---------------------------
# Helper Functions
# ---------------------------
def load_npz_data(path):
    """
    Load data from an npz file and convert to torch tensors.
    Assumes the file contains arrays "X" and "y".
    """
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def get_param_stats(model):
    """
    Return a dictionary with parameter counts for each module in the model.
    Only counts parameters that require gradients.
    """
    stats = {}
    for name, module in model.named_modules():
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if param_count > 0:
            stats[name] = param_count
    return stats


def compute_map(y_true, y_scores, num_classes):
    """
    Compute mAP (mean Average Precision) for multi-class classification.
    Args:
        y_true: numpy array (n_samples,) containing true class indices.
        y_scores: numpy array (n_samples, num_classes) containing predicted probabilities.
        num_classes: total number of classes.
    Returns:
        Mean Average Precision over all classes.
    """
    y_true_onehot = np.zeros_like(y_scores)
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1
    APs = []
    for c in range(num_classes):
        ap = average_precision_score(y_true_onehot[:, c], y_scores[:, c])
        APs.append(ap)
    return np.mean(APs)


# ---------------------------
# Main Training Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a CNN-based network security detection model.")
    # YAML 文件配置参数
    parser.add_argument('--data_yaml',default="E:/cnn_based_network_security_detection_model/Data/processed_data/UNSW_NB15.yaml", type=str,
                        help='Path to dataset YAML config file')  # --data_yaml
    parser.add_argument('--model_yaml',default="E:/cnn_based_network_security_detection_model/models/CNN.yaml", type=str,
                        help='Path to model YAML config file')  # --model_yaml
    # 直接设置训练轮次和批次大小
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    # 用于加载预训练权重继续训练（可选）
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (optional)')
    # 指定所有训练后产生的输出文件保存的目录
    parser.add_argument('--output_dir', type=str, default='run', help='Directory to save all outputs')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)


    # ---------------------------
    # Load YAML Configurations
    # ---------------------------
    with open(args.data_yaml, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model_yaml, 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)

    # ---------------------------
    # Data Loading
    # ---------------------------
    dataset_path = data_cfg['path']  # 数据根目录（预处理后的数据所在文件夹）
    train_file = data_cfg['train']  # 训练集文件，如 "train_split.npz"
    val_file = data_cfg.get('val', None)  # 验证集文件（如果有）
    test_file = data_cfg.get('test', None)  # 测试集文件（用于评估，可选）
    num_classes = data_cfg.get('nc', 10)  # 类别数
    batch_size = args.batch_size  # 使用命令行参数设置的批次大小

    # 加载训练数据
    train_path = os.path.join(dataset_path, train_file)
    X_train, y_train = load_npz_data(train_path)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 加载验证数据（如果存在）
    if val_file:
        val_path = os.path.join(dataset_path, val_file)
        X_val, y_val = load_npz_data(val_path)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    # 加载测试数据（如果提供，用于评估）
    if test_file:
        test_path = os.path.join(dataset_path, test_file)
        X_test, y_test = load_npz_data(test_path)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    # ---------------------------
    # Model Construction
    # ---------------------------
    model_module_path = model_cfg.get('model_module')
    model_class_name = model_cfg.get('model_class')
    model_params = model_cfg.get('params', {})
    if not model_module_path or not model_class_name:
        raise ValueError("Model YAML must specify 'model_module' and 'model_class'.")

    # 动态导入你的模型模块和类
    model_module = importlib.import_module(model_module_path)
    ModelClass = getattr(model_module, model_class_name)
    print("Model parameters:", model_params)
    model = ModelClass(**model_params)

    # ---------------------------
    # Device Setup
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------------------------
    # Resume Training (if specified)
    # ---------------------------
    if args.resume is not None:
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)

    # ---------------------------
    # Define Loss, Optimizer and Training Hyperparameters
    # ---------------------------
    training_cfg = model_cfg.get('training', {})
    learning_rate = training_cfg.get('learning_rate', 0.001)  # 学习率从模型 YAML 中读取
    epochs = args.epochs  # 使用命令行参数设置的训练轮次
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
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
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_epoch_loss:.4f}")
            # 保存验证集上表现最好的模型权重到 output_dir 中
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {epoch_loss:.4f}")

    # 保存最终模型权重到 output_dir
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))

    # ---------------------------
    # 如果测试集存在，则对测试集进行评估，并保存评价指标到 output_dir/test_metrics.txt
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
        probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        test_map = compute_map(all_targets, probs, num_classes)
        conf_mat = confusion_matrix(all_targets, predictions)

        total_samples = len(test_loader.dataset)
        inference_time = end_time - start_time
        speed = total_samples / inference_time

        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test mAP: {test_map:.4f}")
        print(f"Inference Speed: {speed:.2f} samples/second")
        print("Test Confusion Matrix:")
        print(conf_mat)

        with open(os.path.join(args.output_dir, "test_metrics.txt"), "w") as f:
            f.write(f"Test Accuracy: {acc:.4f}\n")
            f.write(f"Test Precision: {precision:.4f}\n")
            f.write(f"Test Recall: {recall:.4f}\n")
            f.write(f"Test F1 Score: {f1:.4f}\n")
            f.write(f"Test mAP: {test_map:.4f}\n")
            f.write(f"Inference Speed: {speed:.2f} samples/second\n")
            f.write("Test Confusion Matrix:\n")
            f.write(np.array2string(conf_mat))

    # ---------------------------
    # 对验证集进行最终评估，并保存评价指标到 output_dir/val_metrics.txt
    # 使用保存的最佳模型权重进行评估
    # ---------------------------
    if val_loader is not None:
        print("Evaluating on validation set using best model weights...")
        best_model = ModelClass(**model_params)
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
        acc = accuracy_score(all_targets, predictions)
        precision = precision_score(all_targets, predictions, average='macro')
        recall = recall_score(all_targets, predictions, average='macro')
        f1 = f1_score(all_targets, predictions, average='macro')
        probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        val_map = compute_map(all_targets, probs, num_classes)
        conf_mat = confusion_matrix(all_targets, predictions)

        total_samples = len(val_loader.dataset)
        inference_time = end_time - start_time
        speed = total_samples / inference_time

        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Validation mAP: {val_map:.4f}")
        print(f"Validation Inference Speed: {speed:.2f} samples/second")
        print("Validation Confusion Matrix:")
        print(conf_mat)

        with open(os.path.join(args.output_dir, "val_metrics.txt"), "w") as f:
            f.write(f"Validation Accuracy: {acc:.4f}\n")
            f.write(f"Validation Precision: {precision:.4f}\n")
            f.write(f"Validation Recall: {recall:.4f}\n")
            f.write(f"Validation F1 Score: {f1:.4f}\n")
            f.write(f"Validation mAP: {val_map:.4f}\n")
            f.write(f"Validation Inference Speed: {speed:.2f} samples/second\n")
            f.write("Validation Confusion Matrix:\n")
            f.write(np.array2string(conf_mat))

    # ---------------------------
    # 保存模型各模块的参数统计信息
    # ---------------------------
    param_stats = get_param_stats(model)
    with open(os.path.join(args.output_dir, "param_stats.txt"), "w") as f:
        for module_name, count in param_stats.items():
            f.write(f"{module_name}: {count} parameters\n")

    # ---------------------------
    # 绘制并保存训练/验证损失曲线到 output_dir/loss_curve.png
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
