"""
高级集成模型训练模块

实现多模型集成训练，包括：

1. 多种基模型：
   - XGBoost: 3个不同随机种子 + 1个深度配置
   - LightGBM: 3个不同随机种子 + 1个深度配置
   - 总计8个基模型

2. 集成策略：
   - 5折交叉验证训练
   - 简单平均融合
   - 基于AUC的加权平均融合
   - 自动选择最佳融合方法

3. 性能优化：
   - 优化的超参数配置
   - 早停机制
   - 并行训练

目标：通过模型多样性和融合策略提升预测性能

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """集成模型训练器，使用多个随机种子和不同配置"""

    def __init__(self, config):
        """
        初始化集成训练器

        参数:
            config (dict): 包含模型参数的配置字典
        """
        self.config = config
        self.models = []
        self.oof_predictions = []
        self.test_predictions = []
        self.feature_importances = None

    def create_model_configs(self):
        """创建多个模型配置"""
        configs = []

        # 不同随机种子的XGBoost配置（减少数量以加快训练）
        for seed in [42, 123, 456]:
            configs.append(
                {
                    "name": f"xgb_seed_{seed}",
                    "model": XGBClassifier,
                    "params": {
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "max_depth": 6,
                        "learning_rate": 0.02,
                        "n_estimators": 1500,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "min_child_weight": 3,
                        "reg_alpha": 0.1,
                        "reg_lambda": 0.1,
                        "random_state": seed,
                        "n_jobs": -1,
                    },
                }
            )

        # 不同随机种子的LightGBM配置（减少数量以加快训练）
        for seed in [42, 123, 456]:
            configs.append(
                {
                    "name": f"lgb_seed_{seed}",
                    "model": LGBMClassifier,
                    "params": {
                        "objective": "binary",
                        "metric": "auc",
                        "boosting_type": "gbdt",
                        "num_leaves": 31,
                        "learning_rate": 0.02,
                        "feature_fraction": 0.9,
                        "bagging_fraction": 0.8,
                        "bagging_freq": 5,
                        "min_child_samples": 20,
                        "reg_alpha": 0.1,
                        "reg_lambda": 0.1,
                        "verbose": -1,
                        "random_state": seed,
                        "n_estimators": 1500,
                        "n_jobs": -1,
                    },
                }
            )

        # 不同超参数的XGBoost配置
        configs.append(
            {
                "name": "xgb_deep",
                "model": XGBClassifier,
                "params": {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "max_depth": 8,
                    "learning_rate": 0.01,
                    "n_estimators": 3000,
                    "subsample": 0.7,
                    "colsample_bytree": 0.7,
                    "min_child_weight": 5,
                    "reg_alpha": 0.2,
                    "reg_lambda": 0.2,
                    "random_state": 42,
                    "n_jobs": -1,
                },
            }
        )

        # 不同超参数的LightGBM配置
        configs.append(
            {
                "name": "lgb_deep",
                "model": LGBMClassifier,
                "params": {
                    "objective": "binary",
                    "metric": "auc",
                    "boosting_type": "gbdt",
                    "num_leaves": 63,
                    "learning_rate": 0.01,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.7,
                    "bagging_freq": 5,
                    "min_child_samples": 30,
                    "reg_alpha": 0.2,
                    "reg_lambda": 0.2,
                    "verbose": -1,
                    "random_state": 42,
                    "n_estimators": 3000,
                    "n_jobs": -1,
                },
            }
        )

        return configs

    def train_ensemble(self, X, y, test_df):
        """
        训练集成模型

        参数:
            X (DataFrame): 训练特征
            y (Series): 目标变量
            test_df (DataFrame): 测试特征

        返回:
            tuple: (折外预测矩阵, 测试集预测矩阵)
        """
        logger.info("开始训练集成模型")

        # 设置交叉验证
        n_folds = self.config["training"]["n_folds"]
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # 获取模型配置
        model_configs = self.create_model_configs()
        n_models = len(model_configs)

        # 初始化预测矩阵
        oof_preds = np.zeros((X.shape[0], n_models))
        test_preds = np.zeros((test_df.shape[0], n_models))
        feature_importance_df = pd.DataFrame()

        # 训练每个模型
        for model_idx, model_config in enumerate(model_configs):
            logger.info(f"训练模型 {model_idx+1}/{n_models}: {model_config['name']}")

            model_oof = np.zeros(X.shape[0])
            model_test = np.zeros(test_df.shape[0])

            # 交叉验证训练
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                logger.info(f"{model_config['name']} - 折 {fold+1}/{n_folds}")

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # 创建并训练模型
                model = model_config["model"](**model_config["params"])

                if "lgb" in model_config["name"]:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:  # xgb
                    model.fit(
                        X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
                    )

                # 生成预测
                model_oof[val_idx] = model.predict_proba(X_val)[:, 1]
                model_test += model.predict_proba(test_df)[:, 1] / n_folds

                # 计算验证集AUC
                fold_auc = roc_auc_score(y_val, model_oof[val_idx])
                logger.info(f"{model_config['name']} - 折 {fold+1} AUC: {fold_auc:.6f}")

                # 记录特征重要性
                fold_importance = pd.DataFrame()
                fold_importance["feature"] = X.columns
                fold_importance["importance"] = model.feature_importances_
                fold_importance["model"] = model_config["name"]
                fold_importance["fold"] = fold + 1
                feature_importance_df = pd.concat(
                    [feature_importance_df, fold_importance], axis=0
                )

            # 保存模型预测
            oof_preds[:, model_idx] = model_oof
            test_preds[:, model_idx] = model_test

            # 计算模型整体AUC
            model_auc = roc_auc_score(y, model_oof)
            logger.info(f"{model_config['name']} 整体AUC: {model_auc:.6f}")

        # 保存预测和特征重要性
        self.oof_predictions = oof_preds
        self.test_predictions = test_preds
        self.feature_importances = (
            feature_importance_df.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
        )

        return oof_preds, test_preds

    def blend_predictions(self, oof_preds, test_preds, y):
        """
        融合预测结果

        参数:
            oof_preds (array): 折外预测矩阵
            test_preds (array): 测试集预测矩阵
            y (Series): 目标变量

        返回:
            tuple: (融合后的折外预测, 融合后的测试集预测)
        """
        logger.info("开始融合预测结果")

        # 简单平均融合
        simple_avg_oof = np.mean(oof_preds, axis=1)
        simple_avg_test = np.mean(test_preds, axis=1)
        simple_auc = roc_auc_score(y, simple_avg_oof)
        logger.info(f"简单平均融合AUC: {simple_auc:.6f}")

        # 加权平均融合（基于单模型AUC）
        model_aucs = []
        for i in range(oof_preds.shape[1]):
            auc = roc_auc_score(y, oof_preds[:, i])
            model_aucs.append(auc)

        weights = np.array(model_aucs)
        weights = weights / weights.sum()

        weighted_avg_oof = np.average(oof_preds, axis=1, weights=weights)
        weighted_avg_test = np.average(test_preds, axis=1, weights=weights)
        weighted_auc = roc_auc_score(y, weighted_avg_oof)
        logger.info(f"加权平均融合AUC: {weighted_auc:.6f}")

        # 选择最佳融合方法
        if weighted_auc > simple_auc:
            logger.info("选择加权平均融合")
            return weighted_avg_oof, weighted_avg_test
        else:
            logger.info("选择简单平均融合")
            return simple_avg_oof, simple_avg_test

    def generate_submission(
        self, test_preds, test_ids, output_path="ensemble_submission.csv"
    ):
        """
        生成提交文件

        参数:
            test_preds (array): 测试集预测
            test_ids (array): 测试集ID
            output_path (str): 输出文件路径
        """
        logger.info("生成集成提交文件")

        # 创建提交DataFrame
        submission = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_preds})

        # 保存到CSV
        submission.to_csv(output_path, index=False)
        logger.info(f"集成提交文件已保存到 {output_path}")

        return submission
