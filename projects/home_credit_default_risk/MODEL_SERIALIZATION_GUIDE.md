# 模型保存和推理系统指南

## 🎯 概述

本项目现在包含完整的模型保存和推理系统，支持：
- ✅ **自动模型保存**：训练完成后自动保存所有模型
- ✅ **版本管理**：支持模型版本控制和元数据管理
- ✅ **推理管线**：完整的推理流水线，支持单样本和批量预测
- ✅ **生产就绪**：可直接部署到生产环境

## 📁 文件结构

```
outputs/
├── models/                    # 训练好的模型文件
│   ├── lgb_1_lightgbm_v20240718_120000.pkl
│   ├── lgb_2_lightgbm_v20240718_120001.pkl
│   └── ensemble_voting_ensemble_v20240718_120002.pkl
├── metadata/                  # 模型元数据
│   ├── lgb_1_lightgbm_v20240718_120000.json
│   ├── lgb_2_lightgbm_v20240718_120001.json
│   └── ensemble_voting_ensemble_v20240718_120002.json
└── feature_builder.pkl        # 特征工程器
```

## 🔧 核心组件

### 1. ModelSerializer - 模型序列化器
负责模型的保存、加载和版本管理。

```python
from models.serializer import ModelSerializer

serializer = ModelSerializer()

# 保存模型
model_path = serializer.save_model(
    model=trained_model,
    model_name="lgb_1",
    model_type="lightgbm",
    metadata={
        'cv_scores': [0.78, 0.79, 0.77],
        'performance': {'cv_auc': 0.78}
    },
    feature_names=feature_names
)

# 加载模型
model, metadata = serializer.load_model(
    model_name="lgb_1",
    model_type="lightgbm",
    latest=True
)

# 列出所有模型
models = serializer.list_available_models()
```

### 2. InferencePipeline - 推理管线
完整的推理流水线，支持模型加载和预测。

```python
from pipeline.inference_pipeline import InferencePipeline

# 初始化推理管线
pipeline = InferencePipeline()

# 加载模型和特征工程器
pipeline.load_models(latest=True)
pipeline.load_feature_builder()

# 单样本预测
result = pipeline.predict_single({
    'AMT_INCOME_TOTAL': 200000,
    'AMT_CREDIT': 400000,
    'AMT_ANNUITY': 25000
})

# 批量预测
pipeline.batch_predict('test.csv', 'predictions.csv')
```

## 🚀 使用方法

### 1. 训练阶段（自动保存）

运行训练脚本，模型会自动保存：

```bash
python main.py
```

训练完成后，所有模型和元数据会自动保存到 `outputs/` 目录。

### 2. 推理阶段

#### 方法1：命令行工具

```bash
# 批量预测
python predict.py --input test_data.csv --output predictions.csv

# 单样本预测
python predict.py --single --data '{"AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 400000}'

# 指定特定模型
python predict.py --input test_data.csv --output predictions.csv --models lgb_1,lgb_2

# 详细输出
python predict.py --input test_data.csv --output predictions.csv --verbose
```

#### 方法2：编程接口

```python
from pipeline.inference_pipeline import InferencePipeline

# 初始化
pipeline = InferencePipeline()
pipeline.load_models()
pipeline.load_feature_builder()

# 预测
result = pipeline.predict_single(sample_data)
pipeline.batch_predict('input.csv', 'output.csv')
```

#### 方法3：独立序列化器（测试用）

```python
from standalone_serializer import StandaloneModelSerializer

serializer = StandaloneModelSerializer()
models = serializer.list_available_models()
model, metadata = serializer.load_model(latest=True)
```

## 📊 元数据格式

每个保存的模型都包含详细的元数据：

```json
{
  "model_name": "lgb_1",
  "model_type": "lightgbm",
  "version": "v20240718_120000",
  "timestamp": "20240718_120000",
  "format": "pickle",
  "path": "./outputs/models/lgb_1_lightgbm_v20240718_120000.pkl",
  "feature_names": ["feature_1", "feature_2", ...],
  "model_fingerprint": "abc123...",
  "cv_scores": [0.78, 0.79, 0.77, 0.76, 0.80],
  "cv_mean": 0.78,
  "cv_std": 0.015,
  "training_samples": 307511,
  "feature_count": 800,
  "model_params": {...},
  "performance": {
    "cv_auc": 0.78
  }
}
```

## 🔍 模型管理

### 列出所有模型

```python
from models.serializer import ModelSerializer

serializer = ModelSerializer()
models = serializer.list_available_models()

for model in models:
    print(f"{model['model_name']} ({model['model_type']})")
    print(f"  版本: {model['version']}")
    print(f"  性能: AUC {model['performance']['cv_auc']:.4f}")
    print(f"  特征数: {model['feature_count']}")
```

### 按类型筛选

```python
# 只显示LightGBM模型
lgb_models = serializer.list_available_models(model_type='lightgbm')

# 只显示特定名称的模型
specific_models = serializer.list_available_models(model_name='lgb_1')
```

### 删除旧模型

```python
# 删除特定版本的模型
serializer.delete_model(
    model_name='old_model',
    model_type='lightgbm',
    version='v20240101_000000'
)
```

## 🎯 推理结果格式

### 单样本预测结果

```python
{
    'prediction': 0.1234,           # 最终预测结果
    'probability': 0.1234,          # 违约概率
    'individual_predictions': {     # 各模型的预测结果
        'lgb_1': 0.1200,
        'lgb_2': 0.1268,
        'ensemble_voting': 0.1234
    }
}
```

### 批量预测结果

输出CSV文件包含：
- `SK_ID_CURR`: 客户ID（如果输入数据包含）
- `prediction`: 最终预测结果
- `probability`: 违约概率

## ⚡ 性能优化

### 批量预测优化

```python
# 使用批处理提高效率
pipeline.batch_predict(
    input_file='large_dataset.csv',
    output_file='predictions.csv',
    batch_size=5000  # 根据内存调整
)
```

### 模型选择优化

```python
# 只加载需要的模型
pipeline.load_models(
    model_configs=[
        {'name': 'lgb_1', 'type': 'lightgbm'},
        {'name': 'lgb_2', 'type': 'lightgbm'}
    ]
)
```

## 🛠️ 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   FileNotFoundError: 模型文件不存在
   ```
   - 确保已运行训练脚本
   - 检查 `outputs/models/` 目录

2. **特征工程器未找到**
   ```
   特征工程器未加载，请先调用load_feature_builder()
   ```
   - 确保 `outputs/feature_builder.pkl` 存在
   - 重新运行训练脚本

3. **特征不匹配**
   ```
   缺少特征: ['feature_x', 'feature_y']
   ```
   - 确保输入数据包含所有必需特征
   - 检查特征工程流程

### 调试模式

```bash
# 启用详细输出
python predict.py --input test.csv --output pred.csv --verbose

# 查看推理演示
python inference_example.py

# 测试序列化功能
python standalone_serializer.py
```

## 🔄 集成到生产环境

### 1. API服务

```python
from flask import Flask, request, jsonify
from pipeline.inference_pipeline import InferencePipeline

app = Flask(__name__)
pipeline = InferencePipeline()
pipeline.load_models()
pipeline.load_feature_builder()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = pipeline.predict_single(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. 批处理服务

```python
import schedule
import time

def daily_batch_prediction():
    pipeline = InferencePipeline()
    pipeline.load_models(latest=True)
    pipeline.load_feature_builder()
    
    pipeline.batch_predict(
        input_file='/data/daily_applications.csv',
        output_file='/data/daily_predictions.csv'
    )

# 每天凌晨2点运行
schedule.every().day.at("02:00").do(daily_batch_prediction)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 🎉 总结

这个模型保存和推理系统提供了：

1. **完整的生命周期管理**：从训练到部署的完整流程
2. **版本控制**：支持模型版本管理和回滚
3. **高性能推理**：优化的批量预测和单样本预测
4. **生产就绪**：完善的错误处理和监控
5. **易于集成**：简单的API接口，易于集成到现有系统

现在你可以：
- ✅ 训练模型后自动保存
- ✅ 随时加载模型进行推理
- ✅ 管理多个模型版本
- ✅ 部署到生产环境
- ✅ 监控模型性能

这是一个企业级的ML模型管理解决方案！
