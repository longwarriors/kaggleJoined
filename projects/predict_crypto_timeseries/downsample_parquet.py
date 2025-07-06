import numpy as np
import pandas as pd
import os

# 获取项目根目录的绝对路径
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))
PROJECT_ROOT = get_project_root()

# 读取原始数据
raw_data_dir = os.path.join(PROJECT_ROOT, "data/raw/drw-crypto-market-prediction")
train_data_file = os.path.join(raw_data_dir, "train.parquet")
test_data_file = os.path.join(raw_data_dir, "test.parquet")
submission_example_file = os.path.join(raw_data_dir, "sample_submission.csv")
train_df = pd.read_parquet(train_data_file)
test_df = pd.read_parquet(test_data_file)
submission_example_df = pd.read_csv(submission_example_file)

# 查看数据
print(f"训练集大小: {train_df.shape}")
print(f"测试集大小: {test_df.shape}")
print(f"提交示例大小: {submission_example_df.shape}")

# 数值列的基本统计信息
print("\n训练数据统计摘要:")
print(train_df.describe().T.head())

# 有“时间戳”列的表格做时间序列降采样
train_df = train_df.sample(frac=0.1, random_state=42)
print(f"重采样后训练集大小: {train_df.shape}")

# 保存
processed_data_dir = os.path.join(PROJECT_ROOT, "data/processed")
train_df.to_parquet(os.path.join(processed_data_dir, "drw-crypto-train_downsampled.parquet"))
