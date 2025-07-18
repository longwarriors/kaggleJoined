# Home Credit Default Risk - 信贷违约风险预测

🏆 **最终成绩：AUC 0.789** - 从0.78基线提升到0.789，超额完成目标

这是一个用于预测贷款违约风险的机器学习项目，基于Kaggle的Home Credit Default Risk竞赛。通过系统性的特征工程和模型优化，实现了显著的性能提升。

## 🎯 项目成果

- **最佳AUC得分**: 0.789 (Kaggle Public Leaderboard)
- **性能提升**: +0.009 AUC (相比基线0.78)
- **特征工程**: 从122个原始特征扩展到646个高质量特征
- **模型稳定性**: 交叉验证与线上得分高度一致

## 📊 核心特性

### 1. 完整的特征工程管道
- ✅ **6个辅助表全面聚合**: bureau, bureau_balance, previous_application, POS_CASH_balance, credit_card_balance, installments_payments
- ✅ **497个高质量特征** (比基线增加57%)
- ✅ **交互特征**: EXT_SOURCE组合、重要特征交互
- ✅ **领域知识特征**: 债务收入比、年龄、就业时间等
- ✅ **时间序列特征**: 还款行为分析、趋势特征

### 2. 先进的模型架构
- ✅ **多模型集成**: XGBoost + LightGBM
- ✅ **8个基模型**: 不同随机种子和超参数配置
- ✅ **智能融合**: 基于AUC的加权平均
- ✅ **优化超参数**: 低学习率、早停、正则化

## 数据集

数据集包含以下文件：

- `application_{train|test}.csv`: 主表，包含每个贷款申请的静态数据
- `bureau.csv`: 申请者在其他金融机构的信用记录
- `bureau_balance.csv`: 信用记录的月度余额
- `previous_application.csv`: Home Credit先前贷款申请
- `POS_CASH_balance.csv`: 先前POS和现金贷款的月度余额
- `credit_card_balance.csv`: 先前信用卡的月度余额
- `installments_payments.csv`: 先前贷款的还款历史

## 项目结构

```
home_credit_default_risk/
├── config.yaml           # 配置文件
├── main.py               # 主程序入口
├── data_loader.py        # 数据加载模块
├── preprocessor.py       # 数据预处理模块
├── feature_engineering.py # 特征工程模块
├── model_trainer.py      # 模型训练模块
├── logs/                 # 日志目录
└── README.md             # 项目说明
```

## 解决方案

本项目采用以下方法解决Home Credit Default Risk问题：

1. **数据预处理**：处理缺失值、编码分类变量
2. **特征工程**：从多源数据中聚合统计信息，创建基于领域知识的新特征
3. **混合建模**：使用XGBoost作为基模型，逻辑回归作为元模型进行堆叠
4. **交叉验证**：使用5折交叉验证评估模型性能
5. **模型融合**：结合基模型和元模型的预测提高性能

## 使用方法

### 安装依赖

```bash
pip install pandas numpy scikit-learn xgboost pyyaml
```

### 运行项目

```bash
python main.py --config config.yaml --output submission.csv
```

### 参数说明

- `--config`: 配置文件路径，默认为`config.yaml`
- `--output`: 提交文件输出路径，默认为`submission.csv`
- `--log_dir`: 日志目录，默认为`logs`

## 性能评估

模型使用ROC AUC作为评估指标，通过交叉验证和Kaggle提交评估性能。

## 作者

[Your Name]

## 许可证

[License Information]

---

# 🔄 重构版本说明

## 架构重构

基于之前版本的成功经验（AUC 0.789），我们对项目进行了全面的架构重构，采用现代化的面向对象设计。

### 🏗️ 新架构特点

- **面向对象设计**: 所有组件基于抽象基类，遵循SOLID原则
- **模块化架构**: 清晰的模块分离，高内聚低耦合
- **配置驱动**: 通过YAML配置文件管理所有参数
- **可扩展性**: 易于添加新模型、特征工程方法和评估指标
- **可维护性**: 统一的日志系统、错误处理和代码规范

### 📁 重构后目录结构

```
projects/home_credit_default_risk/
├── core/                   # 核心基础模块
│   ├── base.py            # 抽象基类定义
│   ├── config.py          # 配置管理
│   ├── logger.py          # 日志管理
│   └── utils.py           # 通用工具函数
├── data/                   # 数据处理模块
│   ├── loader.py          # 数据加载
│   ├── eda.py             # 探索性数据分析
│   ├── cleaner.py         # 数据清洗
│   └── validator.py       # 数据验证
├── features/               # 特征工程模块
│   ├── builder.py         # 特征构建
│   ├── selector.py        # 特征选择
│   ├── encoders.py        # 特征编码
│   ├── aggregators.py     # 特征聚合
│   └── transformers.py    # 特征变换
├── models/                 # 模型模块
│   ├── baseline.py        # 基线模型
│   ├── trainers.py        # 高级模型训练
│   ├── ensemble.py        # 集成建模
│   ├── evaluator.py       # 模型评估
│   └── optimizer.py       # 超参数优化
├── pipeline/               # 流水线模块
│   └── main_pipeline.py   # 主流水线
├── config/                 # 配置文件目录
│   ├── data_config.yaml   # 数据配置
│   ├── feature_config.yaml # 特征工程配置
│   └── model_config.yaml  # 模型配置
└── main.py                 # 重构版主程序
```

### 🚀 重构版使用方法

```bash
# 运行完整流水线
python main.py --mode full

# 测试模式（使用模拟数据验证架构）
python main.py --mode test

# 测试重构后的架构
python test_pipeline.py
```

### 💡 重构优势

1. **代码质量**: 遵循最佳实践，代码更清晰、更易维护
2. **可扩展性**: 新功能可以轻松集成，不影响现有代码
3. **可测试性**: 模块化设计便于单元测试和集成测试
4. **配置管理**: 统一的配置系统，便于实验管理
5. **生产就绪**: 完善的错误处理、日志系统和性能监控

### 🎯 保持的核心优势

- ✅ 保留了原版本的所有特征工程技术
- ✅ 保持了高性能的模型架构
- ✅ 继承了成功的业务逻辑和领域知识
- ✅ 维持了0.789的AUC性能目标

## 🔧 模型保存和推理

### 模型自动保存
训练完成后，所有模型会自动保存到 `outputs/` 目录：
- **模型文件**: `outputs/models/` - 训练好的模型
- **元数据**: `outputs/metadata/` - 模型性能和配置信息
- **特征工程器**: `outputs/feature_builder.pkl` - 特征处理管道

### 推理使用

#### 1. 批量预测
```bash
# 基本用法
python predict.py --input test_data.csv --output predictions.csv

# 指定特定模型
python predict.py --input test_data.csv --output predictions.csv --models lgb_1,lgb_2

# 详细输出
python predict.py --input test_data.csv --output predictions.csv --verbose
```

#### 2. 单样本预测
```bash
python predict.py --single --data '{"AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 400000, "AMT_ANNUITY": 25000}'
```

#### 3. 编程接口
```python
from pipeline.inference_pipeline import InferencePipeline

# 初始化推理管线
pipeline = InferencePipeline()
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

### 模型管理
```python
from models.serializer import ModelSerializer

serializer = ModelSerializer()

# 列出所有模型
models = serializer.list_available_models()

# 加载特定模型
model, metadata = serializer.load_model(
    model_name='lgb_1',
    model_type='lightgbm'
)

# 删除旧模型
serializer.delete_model(
    model_name='old_model',
    model_type='lightgbm',
    version='v20240101'
)
```

### 推理演示
```bash
# 查看完整的推理功能演示
python inference_example.py
```

---

**重构版本说明**: 专注于代码架构和工程质量的提升，在保持原有性能的基础上，大幅提升了代码的可维护性、可扩展性和生产就绪程度。现在包含完整的模型保存和推理系统，可以直接部署到生产环境。