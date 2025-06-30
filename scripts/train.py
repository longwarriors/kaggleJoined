import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple


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
                logits = model(X)
                loss = loss_fn(logits, y)
                preds = torch.argmax(logits, dim=-1)
                batch_correct = (preds == y).sum().item()
                total_correct += batch_correct
                total_loss += loss.item() * batch_size  # 累积加权损失
                total_samples += batch_size
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
