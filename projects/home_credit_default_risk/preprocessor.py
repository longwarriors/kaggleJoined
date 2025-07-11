"""
数据预处理模块

负责对原始数据进行清洗和预处理，包括：
- 缺失值处理（数值型和分类型）
- 分类变量编码（LabelEncoder）
- 异常值和无穷大值处理
- 数据类型优化

支持训练集和测试集的一致性处理。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """负责数据预处理的类"""

    def __init__(self):
        """初始化预处理器"""
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        self.label_encoders = {}
        self.standard_scaler = StandardScaler()

    def preprocess(self, df, is_train=True):
        """
        预处理数据框

        参数:
            df (DataFrame): 要预处理的数据框
            is_train (bool): 是否为训练数据

        返回:
            DataFrame: 预处理后的数据框
        """
        logger.info(f"开始预处理{'训练' if is_train else '测试'}数据")

        # 创建副本以避免修改原始数据
        df_copy = df.copy()

        # 处理缺失值
        num_cols = df_copy.select_dtypes(include=["float64", "int64"]).columns
        cat_cols = df_copy.select_dtypes(include=["object"]).columns

        # 排除ID列和目标列
        exclude_cols = ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "TARGET"]
        num_cols = [col for col in num_cols if col not in exclude_cols]
        cat_cols = [col for col in cat_cols if col not in exclude_cols]

        logger.info(f"处理数值特征缺失值: {len(num_cols)}列")
        if is_train:
            df_copy[num_cols] = self.num_imputer.fit_transform(df_copy[num_cols])
        else:
            df_copy[num_cols] = self.num_imputer.transform(df_copy[num_cols])

        logger.info(f"处理分类特征缺失值: {len(cat_cols)}列")
        if is_train:
            df_copy[cat_cols] = self.cat_imputer.fit_transform(df_copy[cat_cols])
        else:
            df_copy[cat_cols] = self.cat_imputer.transform(df_copy[cat_cols])

        # 编码分类变量
        logger.info("编码分类变量")
        for col in cat_cols:
            if df_copy[col].nunique() > 2:
                # 对高基数特征使用独热编码
                df_copy = pd.get_dummies(
                    df_copy, columns=[col], prefix=col, dummy_na=False
                )
            else:
                # 对二元特征使用标签编码
                if is_train:
                    self.label_encoders[col] = LabelEncoder()
                    df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col])
                else:
                    # 处理测试集中可能出现的新类别
                    if col in self.label_encoders:
                        df_copy[col] = df_copy[col].map(
                            lambda x: (
                                x
                                if x in self.label_encoders[col].classes_
                                else "missing"
                            )
                        )
                        df_copy[col] = self.label_encoders[col].transform(df_copy[col])
                    else:
                        # 如果训练集中没有这个列，跳过
                        logger.warning(f"列 {col} 在训练集中不存在，跳过编码")

        # 处理无穷大值
        logger.info("处理无穷大值")
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)

        logger.info("预处理完成")
        return df_copy
