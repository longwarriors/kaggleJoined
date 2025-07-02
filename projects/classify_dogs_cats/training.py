import sys
import os


# 添加项目根目录到Python路径的更健壮方法
def get_project_root():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)

    # 向上查找，直到找到包含 'datasets' 和 'scripts' 目录的根目录
    while current_dir != os.path.dirname(current_dir):  # 避免到达文件系统根目录
        if os.path.exists(os.path.join(current_dir, "datasets")) and os.path.exists(
            os.path.join(current_dir, "scripts")
        ):
            return current_dir
        current_dir = os.path.dirname(current_dir)

    # 如果没找到，使用固定的向上三级方法作为备选
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


project_root = get_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ========================================================================================#
#                                   开始算法
# ========================================================================================#

from scripts import (
    save_checkpoint,
    load_checkpoint,
    calculate_class_weights,
    EarlyStopping,
    train_epoch,
    validate_epoch,
)
from datasets.cats_dogs import labelled_dogcat_set
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18, ResNet18_Weights

# 项目参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.8
BATCH_SIZE = 512
NUM_EPOCHS = 50
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints/classify_dogs_cats")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
best_ckpt_file_path = os.path.join(CHECKPOINT_DIR, "best.pth")

# 数据集
labelled_set_size = len(labelled_dogcat_set)

train_size = int(TRAIN_RATIO * labelled_set_size)
valid_size = labelled_set_size - train_size
print(f"训练集大小: {train_size}, 验证集大小: {valid_size}")

train_set, valid_set = random_split(labelled_dogcat_set, [train_size, valid_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(
    valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

class_weights = calculate_class_weights(
    dl_train=train_loader, mode="subsample", device=DEVICE
)
print(f"类别权重: {class_weights}")

# 创建模型
pre_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)

# 查看层的名称和是否需要梯度
for name, param in pre_model.named_parameters():
    print(f"Layer Name: {name}, \nParameters Shape: {param.shape}")
    print("--" * 50)
print(f"最后一层: {pre_model.fc}")

# 拼接最后一层适应二分类
add_in_features = pre_model.fc.out_features  # 1000
pre_model.fc = nn.Sequential(
    pre_model.fc,  # 保留原始 fc 层 (in_features=512, out_features=1000)
    nn.ReLU(),
    nn.Linear(add_in_features, 50),
    nn.ReLU(),
    nn.Linear(50, 2),  # 2 classes
).to(DEVICE)
print(f"修改后的最后一层: {pre_model.fc}")

# 冻结与解冻
for param in pre_model.parameters():
    param.requires_grad = False
for param in pre_model.fc.parameters():
    param.requires_grad = True

# 定义损失函数、优化器、学习率调度器
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, pre_model.parameters()), lr=2e-03
)
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

# 早停
early_stopping = EarlyStopping(patience=7, delta=1e-4, mode="max")

# 断点续训练
if os.path.exists(os.path.join(CHECKPOINT_DIR, "best.pth")):
    start_epoch, best_acc = load_checkpoint(
        best_ckpt_file_path, pre_model, optimizer, None, DEVICE
    )
    print(
        f"加载训练点模型成功，当前准确率为{best_acc:.4f}，从第{start_epoch}个epoch开始训练..."
    )
else:
    start_epoch = 0
    best_acc = 0.0

print("开始训练...")
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 50)
    train_loss, train_acc = train_epoch(
        pre_model, train_loader, optimizer, criterion, DEVICE
    )
    valid_loss, valid_acc = validate_epoch(pre_model, valid_loader, criterion, DEVICE)

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
        best_ckpt = save_checkpoint(
            pre_model,
            optimizer,
            None,
            epoch,
            valid_acc,
            best_ckpt_file_path,
        )
    print(f"最佳验证准确率: {valid_acc:.4f}\n")
