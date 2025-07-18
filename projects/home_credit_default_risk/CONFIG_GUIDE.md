# 配置文件使用指南

## 📁 配置文件概览

项目提供了三种配置方式，适应不同的使用场景：

### 1. 简单配置 (`config.yaml`)
- **用途**: 快速原型开发、简单实验
- **特点**: 轻量级、易于理解
- **包含**: 基本数据路径和模型参数

```yaml
# 数据路径配置
datasets:
  application_train: '../../data/raw/home-credit-default-risk/application_train.csv'
  application_test: '../../data/raw/home-credit-default-risk/application_test.csv'
  # ... 其他数据文件

# 模型参数
xgboost_params:
  max_depth: 6
  learning_rate: 0.02
  # ... 其他参数
```

### 2. 综合配置 (`config_comprehensive.yaml`)
- **用途**: 生产环境、完整项目
- **特点**: 功能完整、高度可配置
- **包含**: 项目信息、环境配置、特征工程、模型配置、监控等

```yaml
# 项目基本信息
project:
  name: "home_credit_default_risk"
  version: "2.0.0"
  description: "Home Credit Default Risk Prediction - 重构版"

# 环境配置
environment:
  python_interpreter: "D:/Anaconda/customenvs/trainer/python.exe"
  random_seed: 42
  log_level: "INFO"

# 数据配置
data:
  paths: {...}
  validation: {...}
  cleaning: {...}
  memory_optimization: {...}

# 特征工程配置
features:
  basic_processing: {...}
  advanced: {...}
  selection: {...}

# 模型配置
models:
  problem_type: "binary_classification"
  cross_validation: {...}
  baseline: {...}
  advanced: {...}
  ensemble: {...}
```

### 3. 模块化配置 (`config/`)
- **用途**: 大型项目、团队协作
- **特点**: 模块分离、易于维护
- **包含**: 分模块的配置文件

```
config/
├── data_config.yaml      # 数据相关配置
├── feature_config.yaml   # 特征工程配置
└── model_config.yaml     # 模型相关配置
```

## 🔧 配置使用方法

### 在代码中加载配置

```python
import yaml

# 方法1: 加载简单配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 方法2: 加载综合配置
with open('config_comprehensive.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 方法3: 使用配置管理器
from core.config import ConfigManager
config_manager = ConfigManager()
configs = config_manager.load_all_configs('config/')
```

### 在训练器中使用配置

```python
class Trainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get_model_params(self, model_name: str):
        # 综合配置格式
        if 'models' in self.config:
            return self.config['models']['advanced'][model_name]['params']
        # 简单配置格式
        else:
            return self.config[f'{model_name}_params']
    
    def get_data_paths(self):
        if 'data' in self.config:
            return self.config['data']['paths']
        else:
            return self.config['datasets']
```

## 📊 配置文件比较

| 特性 | 简单配置 | 综合配置 | 模块化配置 |
|------|----------|----------|------------|
| **文件大小** | 小 (~55行) | 大 (~300行) | 中等 (~100行/文件) |
| **复杂度** | 低 | 高 | 中等 |
| **功能完整性** | 基础 | 完整 | 完整 |
| **维护难度** | 低 | 中等 | 低 |
| **扩展性** | 有限 | 优秀 | 优秀 |
| **团队协作** | 一般 | 一般 | 优秀 |
| **版本控制** | 简单 | 复杂 | 简单 |

## 🎯 使用建议

### 选择指南

1. **开发阶段** → 使用 `config.yaml`
   - 快速实验
   - 参数调试
   - 概念验证

2. **生产环境** → 使用 `config_comprehensive.yaml`
   - 完整功能
   - 详细监控
   - 部署配置

3. **大型项目** → 使用 `config/` 目录
   - 团队协作
   - 模块化开发
   - 分工明确

### 最佳实践

1. **配置分层**
   ```
   默认配置 → 环境配置 → 用户配置 → 运行时配置
   ```

2. **环境变量支持**
   ```yaml
   data:
     paths:
       application_train: ${DATA_PATH}/application_train.csv
   ```

3. **配置验证**
   ```python
   def validate_config(config):
       required_keys = ['data', 'models']
       for key in required_keys:
           if key not in config:
               raise ValueError(f"Missing required config: {key}")
   ```

4. **配置版本化**
   ```yaml
   project:
     config_version: "2.0.0"
     compatibility: ["2.0.x"]
   ```

## 🔄 配置迁移

### 从简单配置到综合配置

```python
def migrate_simple_to_comprehensive(simple_config):
    comprehensive_config = {
        'project': {
            'name': 'home_credit_default_risk',
            'version': '2.0.0'
        },
        'data': {
            'paths': simple_config.get('datasets', {})
        },
        'models': {
            'advanced': {
                'xgboost': {
                    'params': simple_config.get('xgboost_params', {})
                },
                'lightgbm': {
                    'params': simple_config.get('lightgbm_params', {})
                }
            }
        }
    }
    return comprehensive_config
```

## 🛠️ 配置扩展

### 添加新的配置节

```yaml
# 在 config_comprehensive.yaml 中添加
custom_features:
  enable: true
  feature_types:
    - "time_series"
    - "text_analysis"
    - "image_features"
  
  parameters:
    window_size: 24
    embedding_dim: 128
    image_size: [224, 224]
```

### 动态配置更新

```python
def update_config_at_runtime(config, updates):
    """运行时更新配置"""
    for key_path, value in updates.items():
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
    return config

# 使用示例
updates = {
    'models.advanced.lightgbm.params.learning_rate': 0.005,
    'features.advanced.aggregation.enable': False
}
config = update_config_at_runtime(config, updates)
```

## 📝 配置模板

### 新项目配置模板

```yaml
# new_project_config.yaml
project:
  name: "new_ml_project"
  version: "1.0.0"
  type: "classification"  # classification, regression, time_series

data:
  paths:
    train: "./data/train.csv"
    test: "./data/test.csv"
  
  target_column: "target"
  id_column: "id"

features:
  enable: true
  methods: ["basic", "advanced"]

models:
  baseline: ["logistic_regression", "random_forest"]
  advanced: ["lightgbm", "xgboost"]
  
  evaluation:
    metric: "roc_auc"
    cv_folds: 5

output:
  submission_file: "submission.csv"
  model_dir: "./models"
```

## 🔍 配置调试

### 配置验证工具

```python
def debug_config(config, verbose=True):
    """调试配置文件"""
    issues = []
    
    # 检查必需的键
    required_keys = ['data', 'models']
    for key in required_keys:
        if key not in config:
            issues.append(f"Missing required key: {key}")
    
    # 检查数据路径
    if 'data' in config and 'paths' in config['data']:
        for name, path in config['data']['paths'].items():
            if not Path(path).exists():
                issues.append(f"Data file not found: {name} -> {path}")
    
    # 检查模型参数
    if 'models' in config:
        for model_name, model_config in config['models'].get('advanced', {}).items():
            if 'params' not in model_config:
                issues.append(f"Missing params for model: {model_name}")
    
    if verbose:
        if issues:
            print("❌ 配置问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ 配置验证通过")
    
    return len(issues) == 0, issues
```

## 🎉 总结

配置系统是项目架构的重要组成部分，提供了：

1. **灵活性**: 多种配置方式适应不同场景
2. **可维护性**: 清晰的配置结构和文档
3. **可扩展性**: 易于添加新的配置项
4. **通用性**: 可复用到其他ML项目

选择合适的配置方式，能够显著提升开发效率和项目质量！
