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

import torch
import torch.nn as nn
import pandas as pd
from torchvision.models import resnet18, ResNet18_Weights
from scripts import inference_with_ids, create_submission_csv
from datasets.cats_dogs import inference_dogcat_set
from torch.utils.data import DataLoader
from datetime import datetime

print("开始推理测试集...")

# 处理数据集
unlabelled_set_size = len(inference_dogcat_set)
print(f"测试集大小: {unlabelled_set_size}")

# 创建DataLoader，注意shuffle=False以保持顺序
infer_loader = DataLoader(
    inference_dogcat_set, batch_size=512, shuffle=False, num_workers=0
)

# 加载模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 重新创建模型结构（与训练时相同）
print("创建模型结构...")
pre_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)
add_in_features = pre_model.fc.out_features  # 1000
pre_model.fc = nn.Sequential(
    pre_model.fc,  # 保留原始 fc 层 (in_features=512, out_features=1000)
    nn.ReLU(),
    nn.Linear(add_in_features, 20),
    nn.ReLU(),
    nn.Linear(20, 2),  # 2 classes
).to(DEVICE)

# 加载训练好的权重
model_path = os.path.join(project_root, "checkpoints/classify_dogs_cats/best.pth")
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    print("请确保您已经训练了模型并保存了检查点文件")
    exit(1)

# 加载检查点
checkpoint = torch.load(model_path, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    # 如果是检查点格式
    pre_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"从检查点加载模型权重: {model_path}")
    if "best_acc" in checkpoint:
        print(f"模型最佳准确率: {checkpoint['best_acc']:.4f}")
else:
    # 如果是完整的模型
    pre_model = checkpoint
    print(f"加载完整模型: {model_path}")

# 设置为评估模式
pre_model.eval()

# 推理
print("开始推理...")
all_ids, all_preds = inference_with_ids(pre_model, infer_loader, DEVICE)

print(f"ID和预测结果数量检查:")
print(f"ID数量: {len(all_ids)}")
print(f"预测结果数量: {len(all_preds)}")
print(f"数据集大小: {len(infer_loader.dataset)}")
assert (
    len(all_ids) == len(all_preds) == len(infer_loader.dataset)
), "ID和预测结果数量不匹配!"

print(f"推理完成! 共处理 {len(all_ids)} 张图片")
print(f"预测结果分布: {pd.Series(all_preds).value_counts().to_dict()}")

# 获取当前时间，格式为“年月日时分”
timestamp = datetime.now().strftime("%Y%m%d%H%M")

# 创建提交文件
filename = f"submission_{timestamp}.csv"
output_path = os.path.join(project_root, "projects/classify_dogs_cats", filename)
submission_df = create_submission_csv(all_ids, all_preds, output_path)

print(f"\n提交文件预览:")
print(submission_df.head(10))
print(f"\n完成! 提交文件已保存到: {output_path}")

# 验证提交文件格式
print(f"\n验证提交文件格式:")
print(f"列名: {list(submission_df.columns)}")
print(f"数据类型:")
print(submission_df.dtypes)
print(f"是否有缺失值: {submission_df.isnull().sum().sum()}")
print(f"ID范围: {submission_df['id'].min()} - {submission_df['id'].max()}")
print(f"标签值: {sorted(submission_df['label'].unique())}")
