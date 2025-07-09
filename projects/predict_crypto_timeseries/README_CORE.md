# 加密货币时间序列预测项目

本项目实现了加密货币市场数据的时间序列预测，使用多种机器学习方法。

## 🎯 项目目标

预测加密货币市场的时间序列数据，目标是在Kaggle竞赛中获得高皮尔逊相关系数。

## 📊 性能进展

| 版本 | Kaggle分数 | 验证分数 | 策略 | 状态 |
|------|-----------|----------|------|------|
| 复杂特征工程 | 0.04497 | 0.4779 | 过度复杂 | ❌ 过拟合 |
| **极简线性模型** | **0.068** | **0.1396** | **纯线性回归** | **✅ 成功** |

## 🔧 核心文件

### 主要训练脚本
- `ultra_simple_training.py` - **极简稳定训练**（最佳性能）
- `lstm_enhanced_training.py` - LSTM增强训练
- `lstm_model.py` - LSTM模型实现

### 工具文件
- `data_preprocessing.py` - 数据预处理工具
- `gradient_boosting_models.py` - 梯度提升模型
- `model_evaluation.py` - 模型评估工具
- `prediction_submission.py` - 预测提交生成

## 🚀 使用方法

### 1. 极简训练（推荐）

```bash
python ultra_simple_training.py
```

- 使用15个最重要的原始特征
- 纯Ridge回归模型
- 避免过拟合，泛化能力强

### 2. LSTM增强训练

```bash
python lstm_enhanced_training.py
```

- 结合LSTM和Ridge回归
- 20个特征，时间序列建模
- 更高的验证性能

## 🎯 关键经验

### ✅ 成功策略

1. **极简特征选择**: 只使用最重要的原始特征
2. **纯线性模型**: Ridge回归避免过拟合
3. **稳定预测**: 确保训练集和测试集一致性
4. **简单集成**: 多个Ridge模型的简单平均

### ❌ 失败教训

1. **复杂特征工程**: 导致严重过拟合
2. **深度模型**: 在小数据集上不稳定
3. **过度优化**: 验证集性能高但实际性能差

## 📈 模型性能

### 极简线性模型
- **验证相关系数**: 0.1396
- **Kaggle分数**: 0.068
- **特征数**: 15个原始特征
- **模型**: 3个Ridge回归集成

### LSTM增强模型
- **验证相关系数**: 0.1518
- **特征数**: 20个
- **模型**: Ridge + LSTM集成

## 🔑 核心代码示例

### 极简特征选择

```python
def select_raw_features_only(X, y, n_features=15):
    """只选择原始特征，不做任何工程"""
    correlations = {}
    for col in X.columns:
        try:
            x_col = X[col].fillna(X[col].median())
            y_clean = y.fillna(y.median())
            corr, _ = pearsonr(x_col, y_clean)
            if not np.isnan(corr):
                correlations[col] = abs(corr)
            else:
                correlations[col] = 0
        except:
            correlations[col] = 0
    
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    selected_features = [col for col, _ in sorted_features[:n_features]]
    
    return X[selected_features].copy(), selected_features
```

### Ridge回归集成

```python
def train_ultra_simple_models(X_train, y_train, X_val, y_val):
    """训练极简模型"""
    results = {}
    
    # 多个Ridge回归
    ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    for alpha in ridge_alphas:
        ridge_model = Ridge(alpha=alpha, random_state=42)
        ridge_model.fit(X_train, y_train)
        
        ridge_pred = ridge_model.predict(X_val)
        ridge_corr, _ = pearsonr(y_val, ridge_pred)
        
        results[f'ridge_{alpha}'] = {
            'model': ridge_model, 
            'val_corr': ridge_corr, 
            'predictions': ridge_pred
        }
    
    # 简单平均集成
    top3_models = dict(sorted(results.items(), key=lambda x: x[1]['val_corr'], reverse=True)[:3])
    ensemble_pred = np.zeros_like(y_val)
    for model_info in top3_models.values():
        ensemble_pred += model_info['predictions']
    ensemble_pred /= len(top3_models)
    
    return results, ensemble_pred
```

## 📝 项目总结

这个项目的最大收获是：**简单稳定的方法往往比复杂的特征工程更有效**。

通过回归到最基础的线性模型和原始特征，我们获得了比复杂深度学习模型更好的实际性能。这说明在时间序列预测中，避免过拟合比追求高验证分数更重要。

## 🎉 成功要素

1. **数据理解**: 深入理解训练集和测试集的分布差异
2. **模型选择**: 选择适合数据规模的简单模型
3. **特征工程**: 保持简单，避免过度工程
4. **验证策略**: 关注实际性能而非验证性能
5. **迭代改进**: 基于实际反馈持续优化
