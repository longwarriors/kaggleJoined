from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Optional


class DogCatDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.img_paths = []
        self.labels = []

        for filename in os.listdir(root_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(self.root_dir, filename)
                self.img_paths.append(full_path)
                if is_train:
                    # 从文件名提取标签：cat 为 0，dog 为 1
                    label = 0 if filename.lower().startswith("cat") else 1
                    self.labels.append(label)
                else:
                    label = -1  # 测试集没有标签，使用 -1 作为占位符
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """根据索引获取图像和标签"""
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            # 提取图像 ID（不含扩展名）
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            return image, img_id


# 创建数据集和加载器
transform = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "inference": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


# 获取项目根目录的绝对路径
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


PROJECT_ROOT = get_project_root()

labelled_dogcat_set = DogCatDataset(
    root_dir=os.path.join(
        PROJECT_ROOT, "data/raw/kaggle-dogs-vs-cats-redux-kernels-edition/train"
    ),
    transform=transform["train"],
    is_train=True,
)
inference_dogcat_set = DogCatDataset(
    root_dir=os.path.join(
        PROJECT_ROOT, "data/raw/kaggle-dogs-vs-cats-redux-kernels-edition/test"
    ),
    transform=transform["inference"],
    is_train=False,
)

if __name__ == "__main__":
    from collections import Counter

    labels = [labelled_dogcat_set[i][1] for i in range(len(labelled_dogcat_set))]
    print(Counter(labels))
