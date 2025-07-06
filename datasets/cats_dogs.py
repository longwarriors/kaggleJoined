import os
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
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
    "train": transforms.Compose([
    transforms.Resize(256),                           # 保证图片足够大
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # 随机裁剪并缩放（模拟不同拍摄距离）
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),             # 加一点点竖直翻转扰动
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
]),
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

# https://claude.ai/chat/2bd3c660-6cb3-4b4a-af1b-55a881069c15
PROJECT_ROOT = get_project_root()
# TRAIN_RATIO = 0.65
labelled_dogcat_set = DogCatDataset(
    root_dir=os.path.join(
        PROJECT_ROOT, "data/raw/kaggle-dogs-vs-cats-redux-kernels-edition/train"
    ),
    transform=transform["train"],
    is_train=True,
)
# labelled_set_size = len(labelled_dogcat_set)
# all_indices = list(range(labelled_set_size))
# train_idx, valid_idx = train_test_split(all_indices, train_size=TRAIN_RATIO, random_state=42)
# train_set = Subset(labelled_dogcat_set, train_idx)
# valid_set = Subset(labelled_dogcat_set, valid_idx)

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
