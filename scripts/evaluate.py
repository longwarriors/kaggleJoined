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
    """带ID的推理函数，返回ID列表和对应的预测结果

    注意：此函数要求DataLoader的shuffle=False，否则ID和预测结果无法正确对应
    """
    model.eval()

    # 检查DataLoader是否被shuffle
    if hasattr(dl_inference, "sampler") and hasattr(dl_inference.sampler, "shuffle"):
        if dl_inference.sampler.shuffle:
            raise ValueError("DataLoader不能使用shuffle=True，否则ID和预测结果无法对应")

    # 从数据集的img_paths中提取所有文件名作为ID
    filenames = [os.path.basename(path) for path in dl_inference.dataset.img_paths]
    all_ids = [os.path.splitext(filename)[0] for filename in filenames]  # 去掉扩展名
    all_preds = []

    with torch.no_grad():
        with tqdm(dl_inference, desc="推理中", leave=False) as pbar_inference:
            for batch_idx, (X, *_) in enumerate(pbar_inference):
                X = X.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                pbar_inference.set_postfix({"batch": f"{batch_idx + 1}"})

    return all_ids, all_preds


def inference_with_ids_robust(
    model, dl_inference: DataLoader, device
) -> Tuple[List[str], List[int]]:
    """更健壮的带ID推理函数，支持shuffle的DataLoader

    通过在推理过程中跟踪样本索引来确保ID和预测结果正确对应
    """
    model.eval()

    # 预先提取所有ID
    filenames = [os.path.basename(path) for path in dl_inference.dataset.img_paths]
    all_ids_map = {
        i: os.path.splitext(filename)[0] for i, filename in enumerate(filenames)
    }

    # 存储结果，使用字典来保持索引对应关系
    results = {}

    with torch.no_grad():
        with tqdm(dl_inference, desc="推理中", leave=False) as pbar_inference:
            for batch_idx, batch_data in enumerate(pbar_inference):
                X = batch_data[0]  # 图像数据
                X = X.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()

                # 获取当前batch的样本索引
                if hasattr(dl_inference, "batch_sampler") and hasattr(
                    dl_inference.batch_sampler, "sampler"
                ):
                    # 对于有sampler的情况，需要从sampler中获取真实索引
                    batch_size = len(preds)
                    start_idx = batch_idx * dl_inference.batch_size
                    batch_indices = list(range(start_idx, start_idx + batch_size))

                    # 如果是最后一个batch，可能不满batch_size
                    if start_idx + batch_size > len(dl_inference.dataset):
                        batch_indices = list(
                            range(start_idx, len(dl_inference.dataset))
                        )
                else:
                    # 简单情况：假设按顺序处理
                    batch_size = len(preds)
                    start_idx = batch_idx * dl_inference.batch_size
                    batch_indices = list(range(start_idx, start_idx + batch_size))

                # 存储结果
                for i, pred in enumerate(preds):
                    if i < len(batch_indices):
                        idx = batch_indices[i]
                        results[idx] = (all_ids_map[idx], pred)

                pbar_inference.set_postfix({"batch": f"{batch_idx + 1}"})

    # 按索引顺序整理结果
    sorted_results = [results[i] for i in sorted(results.keys())]
    all_ids = [item[0] for item in sorted_results]
    all_preds = [item[1] for item in sorted_results]

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
