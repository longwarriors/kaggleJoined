"""
DRW 加密市场预测比赛 - 数据预处理模块
处理大数据集的内存优化加载、预处理、特征缩放和数据分割
"""

import numpy as np
import pandas as pd
import os
import gc
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 获取项目根目录
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

PROJECT_ROOT = get_project_root()

class CryptoDataProcessor:
    """加密货币数据预处理器"""

    def __init__(self, use_downsampled: bool = False, downsample_ratio: float = 0.1):
        """
        初始化数据处理器

        Args:
            use_downsampled: 是否使用降采样数据（用于快速测试）
            downsample_ratio: 降采样比例
        """
        self.use_downsampled = use_downsampled
        self.downsample_ratio = downsample_ratio
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'label'

        # 项目根目录
        self.PROJECT_ROOT = PROJECT_ROOT

        # 数据路径
        self.raw_data_dir = os.path.join(PROJECT_ROOT, "data/raw/drw-crypto-market-prediction")
        self.processed_data_dir = os.path.join(PROJECT_ROOT, "data/processed")

        # 确保处理后数据目录存在
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_data(self, data_type: str = 'train') -> pd.DataFrame:
        """
        加载数据，支持内存优化
        
        Args:
            data_type: 'train', 'test', 'submission'
            
        Returns:
            DataFrame
        """
        print(f"正在加载{data_type}数据...")
        
        if data_type == 'train':
            if self.use_downsampled:
                # 使用预处理的降采样数据
                downsampled_file = os.path.join(self.processed_data_dir, "drw-crypto-train_downsampled.parquet")
                if os.path.exists(downsampled_file):
                    print(f"使用降采样数据: {downsampled_file}")
                    return pd.read_parquet(downsampled_file)
            
            # 加载完整训练数据
            train_file = os.path.join(self.raw_data_dir, "train.parquet")
            df = pd.read_parquet(train_file)
            
            if self.use_downsampled and not os.path.exists(downsampled_file):
                # 创建降采样数据
                print(f"创建降采样数据 (比例: {self.downsample_ratio})")
                df = df.sample(frac=self.downsample_ratio, random_state=42)
                df.to_parquet(downsampled_file)
                print(f"降采样数据已保存: {downsampled_file}")
            
            return df
            
        elif data_type == 'test':
            test_file = os.path.join(self.raw_data_dir, "test.parquet")
            return pd.read_parquet(test_file)
            
        elif data_type == 'submission':
            submission_file = os.path.join(self.raw_data_dir, "sample_submission.csv")
            return pd.read_csv(submission_file)
        
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
    
    def analyze_data(self, df: pd.DataFrame) -> dict:
        """
        分析数据基本信息
        
        Args:
            df: 数据框
            
        Returns:
            分析结果字典
        """
        analysis = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
        }
        
        if self.target_column in df.columns:
            analysis['target_stats'] = df[self.target_column].describe().to_dict()
        
        return analysis
    
    def prepare_features(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        准备特征和目标变量

        Args:
            df: 原始数据框
            is_train: 是否为训练数据

        Returns:
            (特征DataFrame, 目标Series或None)
        """
        print("准备特征和目标变量...")

        # 分离特征和目标
        if is_train and self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            X = df.copy()
            y = None

        # 记录特征列名
        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()

        # 确保特征列一致
        X = X[self.feature_columns]

        # 处理无穷大值和异常值
        print("处理无穷大值和异常值...")

        # 替换无穷大值为NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # 处理NaN值 - 用中位数填充
        if X.isnull().sum().sum() > 0:
            print(f"发现 {X.isnull().sum().sum()} 个缺失值，使用中位数填充")
            X = X.fillna(X.median())

        # 处理极端异常值 - 使用99.9%分位数截断
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                q99 = X[col].quantile(0.999)
                q01 = X[col].quantile(0.001)
                X[col] = X[col].clip(lower=q01, upper=q99)

        # 处理目标变量的异常值
        if y is not None:
            y = y.replace([np.inf, -np.inf], np.nan)
            if y.isnull().sum() > 0:
                print(f"目标变量有 {y.isnull().sum()} 个缺失值，使用均值填充")
                y = y.fillna(y.mean())

        print(f"特征数量: {len(self.feature_columns)}")
        if y is not None:
            print(f"目标变量统计: 均值={y.mean():.4f}, 标准差={y.std():.4f}")

        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_val: Optional[pd.DataFrame] = None, 
                      X_test: Optional[pd.DataFrame] = None, scaler_type: str = 'standard') -> Tuple:
        """
        特征缩放
        
        Args:
            X_train: 训练特征
            X_val: 验证特征
            X_test: 测试特征
            scaler_type: 缩放器类型 ('standard', 'robust')
            
        Returns:
            缩放后的特征
        """
        print(f"使用{scaler_type}缩放器进行特征缩放...")
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的缩放器类型: {scaler_type}")
        
        # 拟合并转换训练数据
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        results = [X_train_scaled]
        
        # 转换验证数据
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            results.append(X_val_scaled)
        
        # 转换测试数据
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            results.append(X_test_scaled)
        
        return tuple(results)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        分割训练和验证数据
        
        Args:
            X: 特征
            y: 目标变量
            test_size: 验证集比例
            random_state: 随机种子
            
        Returns:
            (X_train, X_val, y_train, y_val)
        """
        print(f"分割数据: 训练集{1-test_size:.1%}, 验证集{test_size:.1%}")
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
    
    def create_time_series_data(self, X: pd.DataFrame, y: pd.Series, 
                               window_size: int = 10, step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列数据（滑动窗口）
        
        Args:
            X: 特征数据
            y: 目标变量
            window_size: 窗口大小
            step_size: 步长
            
        Returns:
            (X_sequences, y_sequences)
        """
        print(f"创建时间序列数据: 窗口大小={window_size}, 步长={step_size}")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(0, len(X) - window_size + 1, step_size):
            X_sequences.append(X.iloc[i:i+window_size].values)
            y_sequences.append(y.iloc[i+window_size-1])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"时间序列数据形状: X={X_sequences.shape}, y={y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def save_processed_data(self, data: dict, filename: str):
        """
        保存处理后的数据
        
        Args:
            data: 数据字典
            filename: 文件名
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        
        if filename.endswith('.parquet'):
            # 保存为parquet格式
            for key, df in data.items():
                if isinstance(df, pd.DataFrame):
                    df.to_parquet(filepath.replace('.parquet', f'_{key}.parquet'))
        else:
            # 保存为numpy格式
            np.savez_compressed(filepath, **data)
        
        print(f"处理后数据已保存: {filepath}")
    
    def memory_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        内存优化
        
        Args:
            df: 数据框
            
        Returns:
            优化后的数据框
        """
        print("进行内存优化...")
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # 优化数值类型
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if df[col].dtype == 'int64':
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"内存优化: {initial_memory:.2f}MB -> {final_memory:.2f}MB "
              f"(减少{(initial_memory-final_memory)/initial_memory*100:.1f}%)")
        
        return df


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算皮尔逊相关系数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        皮尔逊相关系数
    """
    return np.corrcoef(y_true, y_pred)[0, 1]


if __name__ == "__main__":
    # 测试数据处理器
    processor = CryptoDataProcessor(use_downsampled=True)
    
    # 加载和分析数据
    train_df = processor.load_data('train')
    analysis = processor.analyze_data(train_df)
    
    print("=== 数据分析结果 ===")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    # 准备特征
    X, y = processor.prepare_features(train_df)
    
    # 分割数据
    X_train, X_val, y_train, y_val = processor.split_data(X, y)
    
    # 特征缩放
    X_train_scaled, X_val_scaled = processor.scale_features(X_train, X_val)
    
    print("\n=== 数据预处理完成 ===")
    print(f"训练集: {X_train_scaled.shape}")
    print(f"验证集: {X_val_scaled.shape}")
    
    # 清理内存
    del train_df, X, y
    gc.collect()
