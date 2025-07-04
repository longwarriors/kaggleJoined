import torch, os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Union, List
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(
    dl_train: DataLoader,
    mode: str = "balanced",
    num_classes: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    max_samples: int = 10000,
) -> torch.Tensor:
    """
    计算不平衡的类别权重。

    Args:
        dl_train (DataLoader): 训练集dataloader，要求返回(batch_x, batch_y)，其中batch_y为标签。
        mode (str): "balanced"（全量）或 "subsample"（子采样，适合大数据集）。
        num_classes (int, optional): 类别数，若为None则自动检测。
        device (str or torch.device): 返回的权重tensor所在设备。
        max_samples (int): 子采样时的最大样本数，mode='subsample'时生效。

    Returns:
        torch.Tensor: 类别权重，shape=[num_classes]，float32。
    """
    all_labels = []
    total_samples = 0
    for batch_idx, (_, y) in enumerate(dl_train):
        if isinstance(y, torch.Tensor):
            batch_labels = y.cpu().numpy().flatten()
        else:
            batch_labels = np.array(y).flatten()
        all_labels.extend(batch_labels)
        total_samples += len(batch_labels)

        # 对于balanced模式，如果数据集太大，可以考虑提前停止
        if mode == "balanced" and total_samples >= max_samples:
            print(f"警告: 数据集很大({total_samples}样本)，考虑使用'subsample'模式")

    all_labels = np.array(all_labels, dtype=np.int64)
    unique_labels = np.unique(all_labels)

    # 验证标签是否从0开始连续
    expected_labels = np.arange(len(unique_labels))
    if not np.array_equal(unique_labels, expected_labels):
        print(f"警告: 标签不连续，发现标签: {unique_labels}，期望: {expected_labels}")

    # 自动检测类别数
    if num_classes is None:
        num_classes = len(unique_labels)
    elif len(unique_labels) != num_classes:
        raise ValueError(
            f"类别数不符: 数据中有{len(unique_labels)}类, 而num_classes={num_classes}"
        )

    # 计算类别权重
    if mode == "balanced":
        weights = compute_class_weight(
            class_weight="balanced", classes=unique_labels, y=all_labels
        )
    elif mode == "subsample":
        idx = np.random.choice(
            len(all_labels), size=min(max_samples, len(all_labels)), replace=False
        )
        subsample_labels = all_labels[idx]
        weights = compute_class_weight(
            class_weight="balanced", classes=unique_labels, y=subsample_labels
        )
    else:
        raise ValueError(
            f"未知类别权重模式: {mode}，支持的模式: 'balanced', 'subsample'"
        )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_acc": best_acc,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    print(f"Checkpoint loaded from {path}, resume from epoch {start_epoch}")
    return start_epoch, best_acc


class EarlyStopping:
    """早停模块，用于防止过拟合"""

    def __init__(
        self,
        patience: int = 7,
        delta: float = 1e-4,
        mode: str = "min",
    ):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_score: float):
        if self.mode == "min":  # 分数越低越好（如损失）
            is_improved = (
                self.best_score is None or metric_score < self.best_score - self.delta
            )
        elif self.mode == "max":  # 分数越高越好（如准确率）
            is_improved = (
                self.best_score is None or metric_score > self.best_score + self.delta
            )
        else:
            raise ValueError(f"不支持的模式: {self.mode}，应为 'min' 或 'max'")

        if is_improved:
            self.best_score = metric_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_epoch(
    model, dl_train: DataLoader, optimizer, loss_fn, device
) -> Tuple[float, float]:
    model.train()
    batch_count = len(dl_train)
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    # 使用 with 语句确保 tqdm 正确关闭
    with tqdm(dl_train, desc="训练中", leave=False) as pbar_train:
        for batch_idx, (X, y) in enumerate(pbar_train):
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)

            # 前向传播
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算准确率和累积损失
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                batch_correct = (preds == y).sum().item()
                total_correct += batch_correct
                total_loss += loss.item() * batch_size  # 累积加权损失
                total_samples += batch_size
                current_accuracy = (
                    total_correct / total_samples
                )  # 迭代到当前batch的平均准确率
                pbar_train.set_postfix(
                    {
                        "batch_loss": f"{loss.item():.4f}",
                        "epoch_accuarcy": f"{current_accuracy:.4f}",
                    }
                )
    # 计算整个epoch的平均损失和准确率
    avg_epoch_loss = total_loss / total_samples
    avg_epoch_accuracy = total_correct / total_samples
    return avg_epoch_loss, avg_epoch_accuracy


def validate_epoch(model, dl_valid: DataLoader, loss_fn, device) -> Tuple[float, float]:
    model.eval()
    batch_count = len(dl_valid)
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0
    with torch.no_grad():
        with tqdm(dl_valid, desc="验证中", leave=False) as pbar_valid:
            for batch_idx, (X, y) in enumerate(pbar_valid):
                X, y = X.to(device), y.to(device)
                batch_size = X.size(0)

                # 前向传播
                logits = model(X)
                loss = loss_fn(logits, y)

                # 计算预测结果和准确率
                preds = torch.argmax(logits, dim=-1)
                batch_correct = (preds == y).sum().item()

                # 累积统计
                total_correct += batch_correct
                total_loss += loss.item() * batch_size  # 累积加权损失
                total_samples += batch_size

                # 更新 tqdm 进度条
                current_accuracy = (
                    total_correct / total_samples
                )  # 迭代到当前batch的平均准确率
                pbar_valid.set_postfix(
                    {
                        "batch_loss": f"{loss.item():.4f}",
                        "epoch_accuarcy": f"{current_accuracy:.4f}",
                    }
                )
    avg_epoch_loss = total_loss / total_samples
    avg_epoch_accuracy = total_correct / total_samples
    return avg_epoch_loss, avg_epoch_accuracy


def train_loop_with_resume(
    pre_model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    early_stopping,
    best_ckpt_file_path,
    num_epochs,
    device,
):
    # 训练表征曲线记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }
    # 断点续训练
    if os.path.exists(best_ckpt_file_path):
        start_epoch, best_acc = load_checkpoint(
            best_ckpt_file_path, pre_model, optimizer, scheduler, device
        )
        print(
            f"加载训练点模型成功，当前准确率为{best_acc:.4f}，从第{start_epoch}个epoch开始训练..."
        )
    else:
        start_epoch = 0
        best_acc = 0.0
        print("未找到检查点，从头开始训练...")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        train_loss, train_acc = train_epoch(
            pre_model, train_loader, optimizer, criterion, device
        )
        valid_loss, valid_acc = validate_epoch(
            pre_model, valid_loader, criterion, device
        )
        print(f"当前验证准确率: {valid_acc:.4f}")
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        # 早停
        early_stopping(valid_acc)
        if early_stopping.early_stop:
            print("早停触发!")
            break

        # 学习率调度
        scheduler.step()
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.2e}")

        # 保存最佳模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            print(f"更新最佳验证准确率: {best_acc:.4f}")
            save_checkpoint(
                pre_model, optimizer, scheduler, epoch, valid_acc, best_ckpt_file_path
            )

    return best_acc, history
