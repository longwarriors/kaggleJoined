### 关键点

- **数据预处理**：处理缺失值、编码分类变量、标准化数值特征，确保数据适合建模。
- **特征工程**：从多源数据中聚合统计信息，创建基于领域知识的新特征，并选择最相关的特征。
- **混合建模**：使用XGBoost等基模型生成预测，然后通过逻辑回归作为元模型进行堆叠（stacking），以提高性能。
- **评估与调参**：通过交叉验证优化ROC AUC分数，使用网格搜索或贝叶斯优化调整超参数。
- **模型融合**：结合堆叠和平均预测方法，增强模型鲁棒性。
- **工程化框架**：模块化代码结构、使用配置文件、日志记录和版本控制，确保项目可扩展且易于维护。

### 数据预处理

在Home Credit Default Risk比赛中，数据预处理是确保模型性能的关键步骤。比赛提供了多个数据集，包括申请表、信用记录、历史贷款和消费数据等，数据量超过30万样本。以下是预处理的主要步骤：

- **处理缺失值**：对于数值特征，使用均值或中位数填补缺失值；对于分类特征，使用众数或引入“缺失”作为新类别。
- **编码分类变量**：对标称变量（如职业类型）使用独热编码（one-hot encoding），对有序变量（如教育水平）使用标签编码（label encoding）。对于高基数分类变量，可考虑目标编码（target encoding）以减少维度。
- **标准化数值特征**：对于逻辑回归等对特征尺度敏感的模型，需对数值特征进行标准化或归一化。树模型如XGBoost通常不需要此步骤。

### 特征工程

特征工程是提升模型性能的核心，尤其是在处理多源数据时。比赛数据集包括主表（`application_train/test.csv`）和其他辅助表（如`bureau.csv`、`previous_application.csv`等）。以下是特征工程的关键步骤：

- **聚合辅助表数据**：通过`SK_ID_CURR`（申请者ID）对辅助表进行分组，计算统计量（如均值、总和、计数、最大值、最小值）。例如，从`bureau.csv`中提取每个申请者的历史信用次数、平均信用金额等。
- **创建新特征**：基于领域知识，创建如债务收入比、申请与批准贷款金额差等特征。此外，可从时间序列数据（如`bureau_balance.csv`）中提取趋势或模式。
- **特征选择**：使用相关性分析、树模型的特征重要性或递归特征消除（RFE）选择最相关的特征，以减少过拟合和计算成本。

### 混合建模（堆叠）

混合建模通过堆叠（stacking）结合多种模型的预测，以提高整体性能。在比赛中，许多参赛者使用了XGBoost和逻辑回归的堆叠方法。以下是实现方式：

- **基模型**：训练多个基模型（如XGBoost、LightGBM、随机森林），通过交叉验证生成训练集的折外预测（out-of-fold predictions）和测试集预测，以避免过拟合。
- **元模型**：使用逻辑回归作为元模型，将基模型的预测作为输入特征进行训练。逻辑回归因其简单性和处理相关特征的能力（通过正则化）常被选为元模型。
- **多样性**：确保基模型的多样性（如使用不同特征子集或超参数），以捕捉数据的不同模式。

### 评估与超参数调优

比赛的评估指标是ROC AUC分数，因此模型优化需围绕此指标进行。以下是评估与调参的步骤：

- **交叉验证**：使用k折交叉验证（如5折）评估模型性能，确保结果稳健且不过拟合。
- **超参数调优**：通过网格搜索、随机搜索或贝叶斯优化调整基模型和元模型的超参数。例如，调整XGBoost的树深度、学习率等。
- **性能监控**：记录每次调参的ROC AUC分数，选择最佳模型配置。

### 模型融合

模型融合通过结合多个模型的预测进一步提升性能。以下是常用方法：

- **堆叠**：如上所述，使用逻辑回归作为元模型，结合基模型预测。
- **平均预测**：对多个堆叠模型的预测进行简单平均或加权平均，权重可基于模型在验证集上的表现确定。
- **多层融合**：如第9名团队的方案，使用多层堆叠（六层），结合多种模型（如XGBoost、LightGBM、神经网络等）以提高预测精度。

### 工程化框架

为实现高工程化、模块化、可扩展的项目框架，需遵循以下最佳实践：

- **代码结构**：将代码分为独立模块（如数据加载、预处理、特征工程、模型训练、堆叠、评估），每个模块负责单一功能。
- **配置文件**：使用配置文件（如JSON或YAML）管理超参数、特征列表等，便于实验和调整。
- **日志记录**：实现日志记录，跟踪每个步骤的进展和性能指标。
- **版本控制**：使用Git管理代码版本，便于协作和回溯。
- **可扩展性**：对于大数据集，使用Dask或Spark进行分布式处理；对于模型训练，利用GPU加速（如XGBoost支持的GPU加速）提高效率。

通过以上步骤，可以构建一个高效、模块化且可扩展的机器学习项目框架，适用于Home Credit Default Risk比赛及类似任务。

---

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import logging
import yaml

# 设置日志
logging.basicConfig(level=logging.INFO, filename='pipeline.log')
logger = logging.getLogger(__name__)

# 加载配置文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class HomeCreditPipeline:
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.models = []
        self.meta_model = LogisticRegression()

    def load_data(self):
        """加载所有数据集"""
        for dataset in self.config['datasets']:
            self.data[dataset] = pd.read_csv(self.config['datasets'][dataset])
        logger.info("数据加载完成")

    def preprocess_data(self):
        """预处理主表数据"""
        main_df = self.data['application_train']
        
        # 处理缺失值
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        
        num_cols = main_df.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = main_df.select_dtypes(include=['object']).columns
        
        main_df[num_cols] = num_imputer.fit_transform(main_df[num_cols])
        main_df[cat_cols] = cat_imputer.fit_transform(main_df[cat_cols])
        
        # 编码分类变量
        for col in cat_cols:
            if main_df[col].nunique() > 2:
                main_df = pd.get_dummies(main_df, columns=[col], prefix=col)
            else:
                le = LabelEncoder()
                main_df[col] = le.fit_transform(main_df[col])
        
        self.data['application_train'] = main_df
        logger.info("主表预处理完成")

    def feature_engineering(self):
        """特征工程：聚合辅助表并创建新特征"""
        main_df = self.data['application_train']
        
        # 示例：聚合bureau表
        bureau = self.data['bureau']
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({
            'AMT_CREDIT_SUM': ['mean', 'sum', 'count'],
            'DAYS_CREDIT': ['mean', 'max']
        }).reset_index()
        bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
        
        # 合并到主表
        main_df = main_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        
        # 创建新特征：债务收入比
        main_df['DEBT_TO_INCOME'] = main_df['AMT_CREDIT_SUM_sum'] / main_df['AMT_INCOME_TOTAL']
        
        self.data['application_train'] = main_df
        logger.info("特征工程完成")

    def train_base_models(self):
        """训练基模型并生成折外预测"""
        X = self.data['application_train'].drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = self.data['application_train']['TARGET']
        test = self.data['application_test'].drop(['SK_ID_CURR'], axis=1)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(X.shape[0])
        test_preds = np.zeros(test.shape[0])
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBClassifier(**self.config['xgboost_params'])
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(test)[:, 1] / 5
            
            self.models.append(model)
        
        self.data['oof_preds'] = oof_preds
        self.data['test_preds'] = test_preds
        logger.info("基模型训练完成")

    def train_meta_model(self):
        """训练元模型"""
        X_meta = self.data['oof_preds'].reshape(-1, 1)
        y = self.data['application_train']['TARGET']
        
        self.meta_model.fit(X_meta, y)
        logger.info("元模型训练完成")

    def predict(self):
        """生成最终预测"""
        test_preds = self.data['test_preds'].reshape(-1, 1)
        final_preds = self.meta_model.predict_proba(test_preds)[:, 1]
        
        submission = pd.DataFrame({
            'SK_ID_CURR': self.data['application_test']['SK_ID_CURR'],
            'TARGET': final_preds
        })
        submission.to_csv('submission.csv', index=False)
        logger.info("预测完成并保存")

if __name__ == "__main__":
    pipeline = HomeCreditPipeline(config)
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.feature_engineering()
    pipeline.train_base_models()
    pipeline.train_meta_model()
    pipeline.predict()
```

### 详细分析

#### 数据预处理

在Home Credit Default Risk比赛中，数据预处理是构建高质量模型的基础。比赛提供了多个数据集，包括主表（`application_train/test.csv`）和辅助表（如`bureau.csv`、`bureau_balance.csv`、`previous_application.csv`等），总计超过30万样本。以下是详细的预处理步骤：

- **处理缺失值**：
  - **数值特征**：使用均值或中位数填补缺失值。例如，对于`AMT_INCOME_TOTAL`（总收入），可使用中位数填补以减少异常值的影响。
  - **分类特征**：使用众数填补，或将缺失值视为新类别“missing”。例如，`OCCUPATION_TYPE`（职业类型）可能包含缺失值，可填补为“unknown”。
  - **高级方法**：对于复杂情况，可使用KNN或MICE（多重插补）填补缺失值，以捕捉特征间的关系。

- **编码分类变量**：
  - **独热编码**：对标称变量（如`NAME_CONTRACT_TYPE`）使用独热编码，生成二进制列。
  - **标签编码**：对有序变量（如`EDUCATION_TYPE`）使用标签编码，保留顺序信息。
  - **高基数变量**：对于如`ORGANIZATION_TYPE`等高基数变量，可使用目标编码，将类别替换为目标变量的平均值，减少维度。

- **标准化数值特征**：
  - 对于逻辑回归或神经网络等模型，需对数值特征进行标准化（如使用`StandardScaler`），以确保特征尺度一致。
  - 树模型（如XGBoost、LightGBM）对特征尺度不敏感，可跳过此步骤。

#### 特征工程

特征工程是提升模型性能的关键，尤其是在处理多源数据时。比赛数据集的复杂性要求从辅助表中提取信息并与主表合并。以下是详细步骤：

- **聚合辅助表数据**：
  - **bureau.csv**：包含申请者在其他金融机构的信用记录。按`SK_ID_CURR`分组，计算统计量，如：
    - 信用次数（`count`）
    - 平均信用金额（`AMT_CREDIT_SUM`的`mean`）
    - 总债务（`AMT_CREDIT_SUM_DEBT`的`sum`）
  - **bureau_balance.csv**：包含信用记录的月度余额。需先按`SK_ID_BUREAU`聚合（如计算逾期状态的平均值），再按`SK_ID_CURR`聚合。
  - **previous_application.csv**：包含Home Credit的先前贷款申请。计算如先前申请次数、批准率等。
  - **其他表**：类似地处理`POS_CASH_balance.csv`、`credit_card_balance.csv`和`installments_payments.csv`，提取月度趋势或还款行为特征。

- **创建新特征**：
  - **基于领域知识**：创建如债务收入比（`AMT_CREDIT_SUM / AMT_INCOME_TOTAL`）、申请与批准金额差（`AMT_CREDIT - AMT_APPLICATION`）等特征。
  - **时间序列特征**：从`bureau_balance.csv`等表中提取趋势，如余额变化率。
  - **交互特征**：创建特征交互，如`AMT_INCOME_TOTAL * AMT_CREDIT`。

- **特征选择**：
  - 使用相关性分析剔除高度相关的特征。
  - 利用XGBoost或LightGBM的特征重要性排序，选择前N个重要特征。
  - 可选地，使用递归特征消除（RFE）或基于L1正则化的方法进一步精简特征。

#### 混合建模（堆叠）

堆叠（stacking）是比赛中常用的技术，用于结合多种模型的预测以提高性能。用户提到的“XGBoost + LogisticRegression stacking”表明XGBoost作为基模型，逻辑回归作为元模型。以下是实现细节：

- **基模型训练**：
  - **模型选择**：训练多个基模型，如XGBoost、LightGBM、随机森林等。XGBoost因其高效性和对结构化数据的适应性被广泛使用。
  - **交叉验证**：使用5折交叉验证生成折外预测（out-of-fold predictions），确保训练数据不泄露到验证集。
  - **多样性**：通过使用不同特征子集、超参数或随机种子增加基模型的多样性。例如，训练多个XGBoost模型，每个模型使用不同最大深度或子采样率。

- **元模型训练**：
  - 将基模型的折外预测作为新特征，训练逻辑回归作为元模型。
  - 逻辑回归使用L1或L2正则化（如`C=0.001`）以处理基模型预测可能的高度相关性。
  - 训练元模型时，使用主表的`TARGET`作为目标变量。

- **实现细节**：
  - 保存基模型的训练集和测试集预测到单独文件中，便于后续堆叠。
  - 确保基模型预测的格式一致（如概率值），以便元模型处理。

#### 评估与超参数调优

比赛的评估指标是ROC AUC分数，优化此指标是模型训练的核心目标。以下是评估与调参的详细步骤：

- **交叉验证**：
  - 使用5折或10折交叉验证评估模型性能，确保结果稳健。
  - 使用`StratifiedKFold`保持目标变量的分布一致，特别是在不平衡数据集（如本比赛）中。

- **超参数调优**：
  - **基模型**：对XGBoost调整参数如`max_depth`、`learning_rate`、`n_estimators`等。使用随机搜索（`RandomizedSearchCV`）或贝叶斯优化（如`Optuna`）提高效率。
  - **元模型**：调整逻辑回归的正则化参数`C`和正则化类型（L1或L2）。
  - **工具**：使用`GridSearchCV`进行网格搜索，或`Optuna`进行贝叶斯优化，以在合理时间内找到最佳参数。

- **性能监控**：
  - 记录每次调参的ROC AUC分数，比较本地交叉验证分数与公共排行榜分数的差异，检测过拟合。
  - 使用日志记录工具（如Python的`logging`模块）保存调参结果。

#### 模型融合

模型融合通过结合多个模型的预测进一步提升性能。以下是常用方法：

- **堆叠**：
  - 如上所述，使用逻辑回归作为元模型，结合XGBoost等基模型的预测。
  - 可扩展到多层堆叠，如第9名团队的六层堆叠，结合多种模型（如LightGBM、神经网络等）。

- **平均预测**：
  - 对多个堆叠模型的预测进行简单平均或加权平均。
  - 权重可基于模型在验证集上的ROC AUC分数确定。

- **多层融合**：
  - 在多层堆叠中，第二层或第三层预测可进一步输入到新的元模型中。
  - 例如，第9名团队在最后一天通过多层预测提升了0.001的私有排行榜分数。

#### 工程化框架

为实现高工程化、模块化、可扩展的项目框架，需遵循以下最佳实践：

- **代码结构**：
  - 将代码分为独立模块：
    - `data_loader.py`：加载和合并数据集。
    - `preprocess.py`：处理缺失值和编码特征。
    - `feature_engineering.py`：聚合和创建新特征。
    - `model_training.py`：训练基模型和元模型。
    - `evaluation.py`：评估和调参。
    - `submission.py`：生成最终预测和提交文件。

- **配置文件**：
  - 使用YAML或JSON文件存储超参数、特征列表、文件路径等。例如：

    ```yaml
    datasets:
      application_train: 'data/application_train.csv'
      bureau: 'data/bureau.csv'
    xgboost_params:
      max_depth: 6
      learning_rate: 0.1
      n_estimators: 100
    ```

- **日志记录**：
  - 使用Python的`logging`模块记录每个步骤的进展、性能指标和错误信息。
  - 示例：记录每次交叉验证的ROC AUC分数。

- **版本控制**：
  - 使用Git管理代码版本，创建分支进行实验，便于回溯和协作。

- **可扩展性**：
  - **大数据处理**：对于大型数据集，使用Dask或Spark进行分布式处理，优化内存使用。
  - **高效训练**：利用XGBoost和LightGBM的GPU加速功能，缩短训练时间。
  - **并行化**：使用`joblib`或多进程并行训练多个基模型。

#### 示例代码框架

以下是一个简化的Python代码框架，展示如何实现上述流程：

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import logging
import yaml

# 设置日志
logging.basicConfig(level=logging.INFO, filename='pipeline.log')
logger = logging.getLogger(__name__)

# 加载配置文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class HomeCreditPipeline:
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.models = []
        self.meta_model = LogisticRegression()

    def load_data(self):
        """加载所有数据集"""
        for dataset in self.config['datasets']:
            self.data[dataset] = pd.read_csv(self.config['datasets'][dataset])
        logger.info("数据加载完成")

    def preprocess_data(self):
        """预处理主表数据"""
        main_df = self.data['application_train']
        
        # 处理缺失值
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        
        num_cols = main_df.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = main_df.select_dtypes(include=['object']).columns
        
        main_df[num_cols] = num_imputer.fit_transform(main_df[num_cols])
        main_df[cat_cols] = cat_imputer.fit_transform(main_df[cat_cols])
        
        # 编码分类变量
        for col in cat_cols:
            if main_df[col].nunique() > 2:
                main_df = pd.get_dummies(main_df, columns=[col], prefix=col)
            else:
                le = LabelEncoder()
                main_df[col] = le.fit_transform(main_df[col])
        
        self.data['application_train'] = main_df
        logger.info("主表预处理完成")

    def feature_engineering(self):
        """特征工程：聚合辅助表并创建新特征"""
        main_df = self.data['application_train']
        
        # 示例：聚合bureau表
        bureau = self.data['bureau']
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({
            'AMT_CREDIT_SUM': ['mean', 'sum', 'count'],
            'DAYS_CREDIT': ['mean', 'max']
        }).reset_index()
        bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
        
        # 合并到主表
        main_df = main_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        
        # 创建新特征：债务收入比
        main_df['DEBT_TO_INCOME'] = main_df['AMT_CREDIT_SUM_sum'] / main_df['AMT_INCOME_TOTAL']
        
        self.data['application_train'] = main_df
        logger.info("特征工程完成")

    def train_base_models(self):
        """训练基模型并生成折外预测"""
        X = self.data['application_train'].drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = self.data['application_train']['TARGET']
        test = self.data['application_test'].drop(['SK_ID_CURR'], axis=1)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(X.shape[0])
        test_preds = np.zeros(test.shape[0])
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBClassifier(**self.config['xgboost_params'])
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(test)[:, 1] / 5
            
            self.models.append(model)
        
        self.data['oof_preds'] = oof_preds
        self.data['test_preds'] = test_preds
        logger.info("基模型训练完成")

    def train_meta_model(self):
        """训练元模型"""
        X_meta = self.data['oof_preds'].reshape(-1, 1)
        y = self.data['application_train']['TARGET']
        
        self.meta_model.fit(X_meta, y)
        logger.info("元模型训练完成")

    def predict(self):
        """生成最终预测"""
        test_preds = self.data['test_preds'].reshape(-1, 1)
        final_preds = self.meta_model.predict_proba(test_preds)[:, 1]
        
        submission = pd.DataFrame({
            'SK_ID_CURR': self.data['application_test']['SK_ID_CURR'],
            'TARGET': final_preds
        })
        submission.to_csv('submission.csv', index=False)
        logger.info("预测完成并保存")

if __name__ == "__main__":
    pipeline = HomeCreditPipeline(config)
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.feature_engineering()
    pipeline.train_base_models()
    pipeline.train_meta_model()
    pipeline.predict()
```

#### 数据集描述

以下是比赛数据集的详细描述，基于Kaggle提供的信息：

| 数据集名称                     | 描述                                                                                     | 行数与列数（若提供）                                      | 来源                                                                 |
|-----------------------------------|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------|
| application_{train|test}.csv      | 主表，分为训练（含TARGET，307,511行，122特征）和测试（无TARGET，48,744行，121特征）。每行代表一笔贷款。 | 训练：307,511行，122特征；测试：48,744行，121特征 | [Kaggle数据页面](https://www.kaggle.com/c/home-credit-default-risk/data) |
| bureau.csv                       | 申请者在其他金融机构的先前信用记录，针对样本中的贷款客户。每笔贷款可能对应多行。 | 未指定                                                      | [Kaggle数据页面](https://www.kaggle.com/c/home-credit-default-risk/data) |
| bureau_balance.csv               | 先前信用的月度余额。每行代表一个月的信用历史，总行数=（样本贷款数 *先前信用数* 历史月份数）。 | 未指定                                                      | [Kaggle数据页面](https://www.kaggle.com/c/home-credit-default-risk/data) |
| POS_CASH_balance.csv             | Home Credit先前POS和现金贷款的月度余额快照。每行代表一个月的信用历史。 | 未指定                                                      | [Kaggle数据页面](https://www.kaggle.com/c/home-credit-default-risk/data) |
| credit_card_balance.csv          | Home Credit先前信用卡的月度余额快照。每行代表一个月的信用卡历史。 | 未指定                                                      | [Kaggle数据页面](https://www.kaggle.com/c/home-credit-default-risk/data) |
| previous_application.csv         | 样本中客户在Home Credit的先前贷款申请。每行代表一次先前申请。 | 未指定                                                      | [Kaggle数据页面](https://www.kaggle.com/c/home-credit-default-risk/data) |
| installments_payments.csv        | 样本中Home Credit先前贷款的还款历史。每行代表一次付款或未付款。 | 未指定                                                      | [Kaggle数据页面](https://www.kaggle.com/c/home-credit-default-risk/data) |
| HomeCredit_columns_description.csv | 各数据文件中列的描述。                                    | 未指定                                                      | [Kaggle数据页面](https://www.kaggle.com/c/home-credit-default-risk/data) |

#### 比赛中的成功案例

根据搜索结果，以下是一些高排名团队的策略，提供了宝贵的参考：

- **第9名团队**：使用了六层堆叠，包含约200个折外预测，结合了XGBoost、LightGBM、随机森林、神经网络等多种模型，最终使用高正则化的逻辑回归作为元模型，私有排行榜得分0.80391。
- **第2名团队**：使用了LightGBM拟合神经网络模型的残差，类似于堆叠，但避免了时间分割堆叠的复杂性。还使用了卷积神经网络（CNN）和递归神经网络（RNN），结合了超过10,000个特征。
- **第4%解决方案**：使用了LightGBM模型的集成，重点在于特征工程和数据聚合，未明确提到逻辑回归堆叠。
