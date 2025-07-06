"""
DRW 加密市场预测比赛 - LSTM深度学习模型
构建LSTM时间序列模型，处理滑动窗口数据准备和训练
"""

import numpy as np
import pandas as pd
import os
import gc
import pickle
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 深度学习库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# 自定义模块
from data_preprocessing import CryptoDataProcessor, pearson_correlation

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        初始化数据集
        
        Args:
            X: 特征数据 (samples, sequence_length, features)
            y: 目标数据 (samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM回归模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征数
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 全连接层
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_size, fc_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_size // 2, 1)
        )
    
    def forward(self, x):
        """前向传播"""
        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的输出
        if self.bidirectional:
            # 双向LSTM：连接前向和后向的最后隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # 单向LSTM：取最后一层的隐藏状态
            hidden = hidden[-1]
        
        # 全连接层
        output = self.fc(hidden)
        
        return output.squeeze()


class LSTMTrainer:
    """LSTM模型训练器"""
    
    def __init__(self, window_size: int = 10, step_size: int = 1,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False,
                 device: str = None):
        """
        初始化LSTM训练器
        
        Args:
            window_size: 时间窗口大小
            step_size: 滑动步长
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
            bidirectional: 是否使用双向LSTM
            device: 计算设备
        """
        self.window_size = window_size
        self.step_size = step_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        self.model = None
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_pearson': []}
        
        # 模型保存路径
        self.model_dir = os.path.join(CryptoDataProcessor().processed_data_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备时间序列数据
        
        Args:
            X: 特征数据
            y: 目标数据
            
        Returns:
            (X_sequences, y_sequences)
        """
        print(f"准备时间序列数据: 窗口大小={self.window_size}, 步长={self.step_size}")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(0, len(X) - self.window_size + 1, self.step_size):
            X_sequences.append(X.iloc[i:i+self.window_size].values)
            y_sequences.append(y.iloc[i+self.window_size-1])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"序列数据形状: X={X_sequences.shape}, y={y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def create_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            X_train: 训练特征序列
            y_train: 训练目标
            X_val: 验证特征序列
            y_val: 验证目标
            batch_size: 批次大小
            
        Returns:
            (train_loader, val_loader)
        """
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def create_model(self, input_size: int):
        """创建LSTM模型"""
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        print(f"LSTM模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        pearson = pearson_correlation(np.array(all_targets), np.array(all_preds))
        
        return avg_loss, pearson
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              epochs: int = 100, batch_size: int = 64,
              learning_rate: float = 0.001, patience: int = 10) -> Dict[str, Any]:
        """
        训练LSTM模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            patience: 早停耐心值
            
        Returns:
            训练结果字典
        """
        print("开始训练LSTM模型...")
        
        # 准备序列数据
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq, batch_size
        )
        
        # 创建模型
        input_size = X_train_seq.shape[2]
        self.create_model(input_size)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        # 早停
        best_pearson = -1
        patience_counter = 0
        
        # 训练循环
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_pearson = self.validate(val_loader, criterion)
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_pearson'].append(val_pearson)
            
            # 学习率调度
            scheduler.step(val_pearson)
            
            # 早停检查
            if val_pearson > best_pearson:
                best_pearson = val_pearson
                patience_counter = 0
                # 保存最佳模型
                self.save_checkpoint('best_lstm_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Val Pearson: {val_pearson:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"早停于第{epoch+1}轮，最佳皮尔逊相关系数: {best_pearson:.6f}")
                break
        
        # 加载最佳模型
        self.load_checkpoint('best_lstm_model.pth')
        
        # 最终评估
        final_val_loss, final_val_pearson = self.validate(val_loader, criterion)
        
        results = {
            'best_val_pearson': best_pearson,
            'final_val_pearson': final_val_pearson,
            'final_val_loss': final_val_loss,
            'training_history': self.training_history,
            'total_epochs': epoch + 1
        }
        
        print(f"LSTM训练完成!")
        print(f"最佳验证皮尔逊相关系数: {best_pearson:.6f}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 准备序列数据
        X_seq, _ = self.prepare_sequences(X, pd.Series(np.zeros(len(X))))
        
        # 创建数据加载器
        dataset = TimeSeriesDataset(X_seq, np.zeros(len(X_seq)))
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        # 预测
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        filepath = os.path.join(self.model_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'window_size': self.window_size,
                'step_size': self.step_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional
            }
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 如果模型不存在，需要先创建
        if self.model is None:
            config = checkpoint['model_config']
            # 需要知道input_size，这里假设从检查点中可以推断
            # 实际使用时可能需要额外保存input_size
            raise ValueError("需要先创建模型才能加载检查点")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        print(f"检查点已加载: {filepath}")


if __name__ == "__main__":
    # 测试LSTM模型训练
    print("测试LSTM模型训练...")
    
    # 加载数据
    processor = CryptoDataProcessor(use_downsampled=True)
    train_df = processor.load_data('train')
    
    # 数据预处理
    X, y = processor.prepare_features(train_df)
    X_train, X_val, y_train, y_val = processor.split_data(X, y, test_size=0.3)  # 更大的验证集用于时间序列
    X_train_scaled, X_val_scaled = processor.scale_features(X_train, X_val)
    
    # 训练LSTM模型
    lstm_trainer = LSTMTrainer(window_size=5, hidden_size=64, num_layers=1)  # 较小的模型用于测试
    lstm_results = lstm_trainer.train(
        X_train_scaled, y_train, X_val_scaled, y_val,
        epochs=20, batch_size=32, patience=5
    )
    
    print("LSTM模型训练测试完成!")
