# DRW 加密市场预测比赛解决方案

这是一个完整的机器学习解决方案，用于参加 DRW 加密市场预测比赛。该解决方案实现了数据预处理、梯度提升模型、LSTM深度学习模型、模型集成和预测提交的完整管道。

## 🎯 比赛概述

- **目标**: 预测加密货币未来价格变动
- **评估指标**: 皮尔逊相关系数
- **数据**: 专有生产特征 + 公共市场量统计数据
- **数据规模**: 训练集 525,887 × 896，测试集 538,150 × 896

## 📁 项目结构

```
projects/predict_crypto_timeseries/
├── main.py                     # 主运行脚本
├── data_preprocessing.py       # 数据预处理模块
├── gradient_boosting_models.py # 梯度提升模型 (XGBoost, LightGBM)
├── lstm_model.py              # LSTM深度学习模型
├── model_evaluation.py        # 模型评估与集成
├── prediction_submission.py   # 预测与提交
├── downsample_parquet.py      # 数据降采样工具
└── README.md                  # 说明文档
```

## 🚀 快速开始

### 1. 环境准备

确保安装以下依赖包：

```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn scipy joblib
```

### 2. 数据准备

确保数据文件位于正确位置：
```
data/raw/drw-crypto-market-prediction/
├── train.parquet
├── test.parquet
└── sample_submission.csv
```

### 3. 运行模式

#### 快速基线模型（推荐开始）
```bash
python main.py --mode baseline --use_downsampled
```

#### 数据分析
```bash
python main.py --mode data_analysis --use_downsampled
```

#### 完整管道
```bash
python main.py --mode full --use_lstm --tune_hyperparams
```

## 📊 功能特性

### 数据预处理
- ✅ 大数据集内存优化
- ✅ 特征缩放 (Standard/Robust Scaler)
- ✅ 数据分割和验证
- ✅ 时间序列数据准备
- ✅ 缺失值处理

### 模型实现
- ✅ **XGBoost** 回归模型
- ✅ **LightGBM** 回归模型  
- ✅ **LSTM** 时间序列模型
- ✅ 超参数调优 (Grid/Random Search)
- ✅ 早停和模型保存

### 模型评估
- ✅ 皮尔逊相关系数（主要指标）
- ✅ RMSE, MAE, R²等回归指标
- ✅ 模型比较和可视化
- ✅ 残差分析

### 模型集成
- ✅ 加权平均集成
- ✅ 堆叠集成
- ✅ 动态权重计算
- ✅ 集成性能评估

## 🔧 详细使用说明

### 命令行参数

```bash
python main.py [OPTIONS]

选项:
  --mode {baseline,full,data_analysis}  运行模式
  --use_downsampled                     使用降采样数据进行快速测试
  --downsample_ratio FLOAT              降采样比例 (默认: 0.1)
  --tune_hyperparams                    进行超参数调优
  --use_lstm                           使用LSTM模型
  --lstm_window_size INT               LSTM时间窗口大小 (默认: 10)
  --ensemble_method {weighted_average,stacking}  集成方法
```

### 使用示例

1. **快速测试** (使用1%数据)
```bash
python main.py --mode baseline --use_downsampled --downsample_ratio 0.01
```

2. **完整训练** (使用全部数据)
```bash
python main.py --mode full --tune_hyperparams --use_lstm
```

3. **只用梯度提升模型**
```bash
python main.py --mode full --tune_hyperparams
```

4. **数据探索**
```bash
python main.py --mode data_analysis
```

## 📈 模型性能

### 预期性能指标
- **XGBoost**: 皮尔逊相关系数 ~0.15-0.25
- **LightGBM**: 皮尔逊相关系数 ~0.15-0.25  
- **LSTM**: 皮尔逊相关系数 ~0.10-0.20
- **集成模型**: 皮尔逊相关系数 ~0.20-0.30

### 训练时间估算
- **降采样 (10%)**: 5-15分钟
- **完整数据**: 1-3小时
- **超参数调优**: 2-6小时

## 💡 优化建议

### 内存优化
- 使用 `--use_downsampled` 进行快速测试
- 数据类型优化 (float64 → float32)
- 分批处理大数据集

### 性能优化
- 调整 `lstm_window_size` (5-20)
- 尝试不同的集成权重方法
- 增加交叉验证折数

### 特征工程
- 添加技术指标特征
- 时间相关特征
- 交互特征

## 🔍 故障排除

### 常见问题

1. **内存不足**
   - 使用 `--use_downsampled`
   - 减小批次大小
   - 关闭其他程序

2. **CUDA错误** (LSTM训练)
   - 检查PyTorch CUDA版本
   - 使用CPU训练: 设置 `device='cpu'`

3. **数据路径错误**
   - 检查数据文件位置
   - 确认文件名正确

### 调试模式
```bash
python main.py  # 运行快速功能测试
```

## 📝 输出文件

### 提交文件
- 位置: `submissions/`
- 格式: `submission_YYYYMMDD_HHMMSS.csv`
- 列: `ID`, `prediction`

### 模型文件
- 位置: `data/processed/models/`
- XGBoost: `xgboost_final.pkl`
- LightGBM: `lightgbm_final.pkl`
- LSTM: `best_lstm_model.pth`

### 结果文件
- 位置: `data/processed/results/`
- 评估结果: `evaluation_results.pkl`

## 🎯 下一步改进

1. **特征工程**
   - 滞后特征
   - 滚动统计特征
   - 技术指标

2. **模型优化**
   - Transformer模型
   - 更复杂的集成策略
   - 贝叶斯优化

3. **验证策略**
   - 时间序列交叉验证
   - 前向验证

## 📞 支持

如有问题，请检查：
1. 数据文件是否正确放置
2. 依赖包是否完整安装
3. 内存是否充足

---

**祝您在比赛中取得好成绩！** 🏆
