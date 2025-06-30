import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List


def inference(model, dl_inference: DataLoader, device) -> List[int]:
    model.eval()
    with torch.no_grad():
        with tqdm(dl_inference, desc="推理中", leave=False) as pbar_inference:
            for batch_idx, (X, _) in enumerate(pbar_inference):
                X = X.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=-1)
                pbar_inference.set_postfix({"batch_idx": batch_idx})
    return preds.tolist()
