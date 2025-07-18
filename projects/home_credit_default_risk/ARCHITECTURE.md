# Home Credit Default Risk - 重构版架构设计

## 核心设计原则

### 面向对象设计原则
- **单一职责原则**：每个类只负责一个功能
- **开闭原则**：对扩展开放，对修改封闭
- **里氏替换原则**：子类可以替换父类
- **接口隔离原则**：使用多个专门的接口
- **依赖倒置原则**：依赖抽象而不是具体实现

### 模块化设计
- **高内聚**：模块内部功能紧密相关
- **低耦合**：模块间依赖最小化
- **可配置性**：通过配置文件管理参数
- **可测试性**：每个模块可独立测试
- **可扩展性**：易于添加新功能

## 项目架构

```
projects/home_credit_default_risk/
├── core/                           # 核心基础模块
│   ├── __init__.py
│   ├── base.py                     # 基础抽象类
│   ├── config.py                   # 配置管理器
│   ├── logger.py                   # 日志管理器
│   └── utils.py                    # 通用工具函数
├── data/                           # 数据处理模块
│   ├── __init__.py
│   ├── loader.py                   # 数据加载器
│   ├── eda.py                      # 探索性数据分析
│   ├── cleaner.py                  # 数据清洗器
│   └── validator.py                # 数据验证器
├── features/                       # 特征工程模块
│   ├── __init__.py
│   ├── builder.py                  # 特征构建器
│   ├── selector.py                 # 特征选择器
│   ├── encoders.py                 # 特征编码器
│   ├── aggregators.py              # 特征聚合器
│   └── transformers.py             # 特征变换器
├── models/                         # 建模模块
│   ├── __init__.py
│   ├── baseline.py                 # 基线模型
│   ├── trainers.py                 # 模型训练器
│   ├── ensemble.py                 # 集成建模器
│   ├── evaluator.py                # 模型评估器
│   └── optimizer.py                # 超参数优化器
├── pipeline/                       # 流水线模块
│   ├── __init__.py
│   ├── feature_pipeline.py         # 特征工程流水线
│   ├── model_pipeline.py           # 建模流水线
│   └── main_pipeline.py            # 主流水线
├── config/                         # 配置文件
│   ├── data_config.yaml            # 数据配置
│   ├── feature_config.yaml         # 特征工程配置
│   ├── model_config.yaml           # 模型配置
│   └── pipeline_config.yaml        # 流水线配置
├── tests/                          # 测试文件
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   └── test_pipeline/
├── docs/                           # 文档
│   ├── api_reference.md
│   ├── user_guide.md
│   └── development_guide.md
├── outputs/                        # 输出文件
│   ├── models/                     # 保存的模型
│   ├── features/                   # 特征文件
│   ├── submissions/                # 提交文件
│   └── reports/                    # 分析报告
├── main.py                         # 主程序入口
└── requirements.txt                # 依赖包
```

## 核心模块设计

### 1. Core模块 - 基础设施

#### base.py - 基础抽象类
```python
class BaseProcessor(ABC):
    """所有处理器的基类"""
    
class BaseModel(ABC):
    """所有模型的基类"""
    
class BaseEvaluator(ABC):
    """所有评估器的基类"""
```

#### config.py - 配置管理
```python
class ConfigManager:
    """统一配置管理器"""
    def load_config(self, config_path: str) -> Dict
    def get_data_config(self) -> Dict
    def get_feature_config(self) -> Dict
    def get_model_config(self) -> Dict
```

### 2. Data模块 - 数据处理

#### loader.py - 数据加载
```python
class DataLoader(BaseProcessor):
    """数据加载器"""
    def load_all_data(self) -> Dict[str, pd.DataFrame]
    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]
    def validate_data(self) -> bool
```

#### eda.py - 探索性数据分析
```python
class EDAAnalyzer(BaseProcessor):
    """EDA分析器"""
    def analyze_data_overview(self) -> Dict
    def analyze_distributions(self) -> Dict
    def analyze_correlations(self) -> Dict
    def analyze_missing_values(self) -> Dict
    def generate_eda_report(self) -> str
```

#### cleaner.py - 数据清洗
```python
class DataCleaner(BaseProcessor):
    """数据清洗器"""
    def handle_missing_values(self) -> pd.DataFrame
    def handle_outliers(self) -> pd.DataFrame
    def convert_data_types(self) -> pd.DataFrame
    def clean_data(self) -> pd.DataFrame
```

### 3. Features模块 - 特征工程

#### builder.py - 特征构建
```python
class FeatureBuilder(BaseProcessor):
    """特征构建器"""
    def build_numerical_features(self) -> pd.DataFrame
    def build_categorical_features(self) -> pd.DataFrame
    def build_temporal_features(self) -> pd.DataFrame
    def build_interaction_features(self) -> pd.DataFrame
    def build_domain_features(self) -> pd.DataFrame
```

#### selector.py - 特征选择
```python
class FeatureSelector(BaseProcessor):
    """特征选择器"""
    def select_by_variance(self) -> List[str]
    def select_by_correlation(self) -> List[str]
    def select_by_mutual_info(self) -> List[str]
    def select_by_model_importance(self) -> List[str]
```

### 4. Models模块 - 建模

#### baseline.py - 基线模型
```python
class BaselineModel(BaseModel):
    """基线模型"""
    def train_simple_models(self) -> Dict
    def establish_baseline(self) -> float
```

#### ensemble.py - 集成建模
```python
class EnsembleModel(BaseModel):
    """集成模型"""
    def voting_ensemble(self) -> np.ndarray
    def averaging_ensemble(self) -> np.ndarray
    def stacking_ensemble(self) -> np.ndarray
```

### 5. Pipeline模块 - 流水线

#### main_pipeline.py - 主流水线
```python
class HomeCreditPipeline:
    """主流水线"""
    def __init__(self, config_path: str)
    def run_eda(self) -> Dict
    def run_feature_engineering(self) -> pd.DataFrame
    def run_modeling(self) -> Dict
    def run_full_pipeline(self) -> Dict
    def generate_submission(self) -> str
```

## 特征工程流程设计

### 阶段1：数据理解与EDA
1. 数据概览（形状、类型、基本统计）
2. 分布分析（直方图、箱线图）
3. 相关性分析（热力图、相关矩阵）
4. 缺失值分析（缺失模式、缺失率）
5. 异常值分析（Z-score、IQR）

### 阶段2：数据清洗与预处理
1. 缺失值处理（删除、填充、预测）
2. 异常值处理（截断、替换、标记）
3. 数据类型转换（数值、类别、日期）
4. 数据验证（完整性、一致性）

### 阶段3：特征构建
1. **数值特征处理**
   - 标准化/归一化
   - 分箱/离散化
   - 数学变换（对数、幂变换）

2. **类别特征编码**
   - 独热编码
   - 标签编码
   - 目标编码
   - 嵌入编码

3. **时间特征工程**
   - 时间拆分（年、月、日、星期）
   - 时间差计算
   - 周期性特征
   - 节假日标记

4. **交叉特征构建**
   - 特征组合
   - 比值特征
   - 差值特征
   - 乘积特征

5. **聚合特征构建**
   - 分组统计
   - 滑窗统计
   - 历史行为聚合

### 阶段4：特征选择
1. **过滤法**
   - 方差选择
   - 相关系数选择
   - 卡方检验
   - 互信息选择

2. **包裹法**
   - 递归特征消除
   - 序列特征选择

3. **嵌入法**
   - L1正则化
   - 树模型特征重要性

## 混合建模流程设计

### 阶段1：基线模型建立
1. 简单模型训练（逻辑回归、决策树）
2. 基线性能确定
3. 评估指标选择
4. 交叉验证设置

### 阶段2：多模型尝试与调优
1. **树集成模型**
   - XGBoost
   - LightGBM
   - CatBoost

2. **其他模型**
   - 支持向量机
   - 神经网络
   - KNN

3. **超参数调优**
   - 网格搜索
   - 随机搜索
   - 贝叶斯优化

### 阶段3：混合建模策略
1. **投票法**
   - 硬投票
   - 软投票

2. **平均法**
   - 简单平均
   - 加权平均

3. **堆叠法**
   - 两层堆叠
   - 多层堆叠
   - 交叉验证堆叠

4. **Bagging/Boosting**
   - 随机森林
   - 梯度提升

### 阶段4：持续迭代优化
1. 错误分析
2. 特征重要性分析
3. 模型性能监控
4. 超参数微调
5. 集成权重优化

## 配置文件设计

### data_config.yaml
```yaml
data_paths:
  train: "path/to/train.csv"
  test: "path/to/test.csv"
  
data_validation:
  check_missing: true
  check_duplicates: true
  
data_cleaning:
  missing_threshold: 0.5
  outlier_method: "iqr"
```

### feature_config.yaml
```yaml
numerical_features:
  scaling_method: "standard"
  binning_strategy: "quantile"
  
categorical_features:
  encoding_method: "target"
  high_cardinality_threshold: 50
  
interaction_features:
  max_combinations: 2
  importance_threshold: 0.01
```

### model_config.yaml
```yaml
baseline_models:
  - "logistic_regression"
  - "decision_tree"
  
advanced_models:
  xgboost:
    max_depth: 6
    learning_rate: 0.02
    n_estimators: 2000
    
ensemble_methods:
  - "voting"
  - "averaging"
  - "stacking"
```

## 使用示例

```python
from pipeline.main_pipeline import HomeCreditPipeline

# 创建流水线
pipeline = HomeCreditPipeline("config/pipeline_config.yaml")

# 运行EDA
eda_results = pipeline.run_eda()

# 运行特征工程
features_df = pipeline.run_feature_engineering()

# 运行建模
model_results = pipeline.run_modeling()

# 运行完整流水线
results = pipeline.run_full_pipeline()

# 生成提交文件
submission_path = pipeline.generate_submission()
```

这个架构设计确保了：
1. **模块化**：每个功能都有独立的模块
2. **可扩展性**：易于添加新的特征工程方法和模型
3. **可维护性**：清晰的代码结构和文档
4. **可测试性**：每个模块都可以独立测试
5. **可配置性**：通过配置文件管理所有参数
