import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
import pandas as pd


def inference(model, dl_inference: DataLoader, device) -> List[int]:
    """原始推理函数，只返回预测结果"""
    model.eval()
    all_preds = []
    with torch.no_grad():
        with tqdm(dl_inference, desc="推理中", leave=False) as pbar_inference:
            for batch_idx, (X, *_) in enumerate(pbar_inference):
                X = X.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                pbar_inference.set_postfix({"batch": f"{batch_idx + 1}"})
    return all_preds


def inference_with_ids(
    model, dl_inference: DataLoader, device
) -> Tuple[List[str], List[int]]:
    """带ID的推理函数，返回ID列表和对应的预测结果"""
    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        with tqdm(dl_inference, desc="推理中", leave=False) as pbar_inference:
            for batch_idx, (X, ids) in enumerate(pbar_inference):
                all_ids.extend(ids)  # 直接来自 Dataset
                X = X.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                pbar_inference.set_postfix({"batch": f"{batch_idx + 1}"})

    return all_ids, all_preds


def create_submission_csv(
    ids: List[str], predictions: List[int], output_path: str = "submission.csv"
) -> pd.DataFrame:
    """创建提交文件"""
    # 将预测结果转换为标签名称（如果需要的话）
    # 0 -> cat, 1 -> dog（根据您的标签定义）
    label_names = {0: 0, 1: 1}  # 如果需要字符串标签，可以改为 {0: 'cat', 1: 'dog'}

    submission_df = pd.DataFrame(
        {"id": ids, "label": [label_names[pred] for pred in predictions]}
    )

    # 确保ID是整数类型（如果是数字ID的话）
    try:
        submission_df["id"] = submission_df["id"].astype(int)
        submission_df = submission_df.sort_values("id")  # 按ID排序
    except ValueError:
        # 如果ID不是数字，保持字符串格式
        pass

    submission_df.to_csv(output_path, index=False)
    print(f"提交文件已保存到: {output_path}")
    print(f"文件包含 {len(submission_df)} 条记录")
    print(f"预测结果分布:")
    print(submission_df["label"].value_counts())

    return submission_df
