# 使用说明 - DRW 加密市场预测比赛

## 🎯 最终版本 - 快速使用

### 1. 环境要求

```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn scipy joblib
```

### 2. 数据准备

确保数据文件位于：
```
data/raw/drw-crypto-market-prediction/
├── train.parquet
├── test.parquet
└── sample_submission.csv
```

### 3. 运行训练

```bash
python standalone_training.py
```

### 4. 预期结果

- **训练时间**: 约9分钟
- **XGBoost性能**: 验证集皮尔逊相关系数 ~0.978
- **LightGBM性能**: 验证集皮尔逊相关系数 ~0.966
- **输出文件**: `submissions/full_training_submission_YYYYMMDD_HHMMSS.csv`

### 5. 提交文件

生成的提交文件格式：
```csv
ID,prediction
1,-0.2802334989066606
2,1.3719688155968115
...
```

## 📁 核心文件说明

- `standalone_training.py` - 最终训练脚本，包含完整训练流程
- `data_preprocessing.py` - 数据预处理，处理无穷大值和缺失值
- `gradient_boosting_models.py` - XGBoost和LightGBM模型实现
- `lstm_model.py` - LSTM深度学习模型
- `model_evaluation.py` - 模型评估和集成
- `prediction_submission.py` - 预测生成和提交文件创建

## 🚀 性能亮点

- **数据规模**: 525,887条训练数据，895个特征
- **异常值处理**: 智能处理超过1100万个无穷大值
- **模型性能**: 验证集皮尔逊相关系数接近0.98
- **训练效率**: 完整训练仅需9分钟
- **提交格式**: 完全符合竞赛要求
