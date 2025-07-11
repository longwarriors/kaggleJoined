"""
数据加载模块

负责加载Home Credit Default Risk竞赛的所有数据文件，包括：
- application_train.csv: 主训练数据
- application_test.csv: 主测试数据
- bureau.csv: 信贷局数据
- bureau_balance.csv: 信贷局余额数据
- previous_application.csv: 历史申请数据
- POS_CASH_balance.csv: POS和现金贷款余额
- credit_card_balance.csv: 信用卡余额数据
- installments_payments.csv: 分期付款数据

作者：Augment Agent
"""

import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器类

    负责加载和管理Home Credit Default Risk竞赛的所有数据文件。
    支持自动数据验证和错误处理。
    """

    def __init__(self, config):
        """
        初始化数据加载器

        参数:
            config (dict): 包含数据路径的配置字典
        """
        self.config = config
        self.data = {}

    def load_data(self):
        """加载所有数据集"""
        for dataset_name, path in self.config["datasets"].items():
            logger.info(f"加载数据集: {dataset_name}")
            try:
                self.data[dataset_name] = pd.read_csv(path)
                logger.info(
                    f"成功加载 {dataset_name}, 形状: {self.data[dataset_name].shape}"
                )
            except Exception as e:
                logger.error(f"加载 {dataset_name} 失败: {str(e)}")
                raise

        return self.data

    def get_train_test_data(self):
        """返回训练集和测试集"""
        if "application_train" not in self.data or "application_test" not in self.data:
            raise ValueError("请先调用load_data()方法加载数据")

        return self.data["application_train"], self.data["application_test"]
