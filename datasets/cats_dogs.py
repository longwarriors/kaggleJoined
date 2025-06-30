from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class DogCatDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        if is_train:
            for filename in os.listdir(root_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    # 从文件名提取标签：cat 为 0，dog 为 1
                    label = 0 if filename.lower().startswith("cat") else 1
                    self.img_paths.append(os.path.join(self.root_dir, filename))
                    self.labels.append(label)
        else:
            for filename in os.listdir(root_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    # 测试集没有标签，所以使用 -1 作为占位符
                    self.img_paths.append(os.path.join(self.root_dir, filename))
                    self.labels.append(-1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """根据索引获取图像和标签"""
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# 创建数据集和加载器
transform = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    "inference": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
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
