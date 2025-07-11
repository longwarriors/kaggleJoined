"""
高级特征工程模块

实现完整的特征工程管道，包括：

1. 辅助表聚合特征：
   - bureau: 信贷局历史数据聚合
   - bureau_balance: 信贷局余额历史聚合
   - previous_application: 历史申请数据聚合
   - POS_CASH_balance: POS和现金贷款聚合
   - credit_card_balance: 信用卡余额聚合
   - installments_payments: 分期付款行为聚合

2. 领域知识特征：
   - 债务收入比、信用额度收入比
   - 年龄和就业时间特征
   - 收入就业年限比等

3. 交互特征：
   - EXT_SOURCE特征组合（均值、标准差、乘积）
   - 重要特征间的乘积、比值、差值交互

性能优化：使用pd.concat避免DataFrame碎片化

作者：Augment Agent
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """负责特征工程的类"""

    def __init__(self, data_dict):
        """
        初始化特征工程器

        参数:
            data_dict (dict): 包含所有数据集的字典
        """
        self.data = data_dict

    def aggregate_bureau(self, main_df):
        """
        聚合bureau表数据

        参数:
            main_df (DataFrame): 主表数据

        返回:
            DataFrame: 添加了bureau聚合特征的主表
        """
        logger.info("聚合bureau表数据")

        if "bureau" not in self.data:
            logger.warning("bureau表不存在，跳过聚合")
            return main_df

        bureau = self.data["bureau"]

        # 数值列聚合
        num_aggregations = {
            "DAYS_CREDIT": ["min", "max", "mean", "var"],
            "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
            "DAYS_CREDIT_UPDATE": ["mean"],
            "CREDIT_DAY_OVERDUE": ["max", "mean"],
            "AMT_CREDIT_MAX_OVERDUE": ["mean"],
            "AMT_CREDIT_SUM": ["max", "mean", "sum"],
            "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
            "AMT_CREDIT_SUM_OVERDUE": ["mean"],
            "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
            "AMT_ANNUITY": ["max", "mean"],
            "CNT_CREDIT_PROLONG": ["sum"],
        }

        # 分类列聚合 - 只计算计数
        cat_aggregations = {}
        for cat in bureau.select_dtypes(include=["object"]).columns:
            if cat != "SK_ID_CURR":  # 排除ID列
                cat_aggregations[cat] = ["count"]

        bureau_agg = bureau.groupby("SK_ID_CURR").agg(
            {**num_aggregations, **cat_aggregations}
        )
        bureau_agg.columns = [
            "_".join(col).strip() for col in bureau_agg.columns.values
        ]

        # 合并到主表
        main_df = main_df.merge(bureau_agg, on="SK_ID_CURR", how="left")

        logger.info(f"bureau聚合完成，添加了{bureau_agg.shape[1]}个特征")
        return main_df

    def aggregate_bureau_balance(self, main_df):
        """聚合bureau_balance表数据"""
        logger.info("聚合bureau_balance表数据")

        if "bureau_balance" not in self.data or "bureau" not in self.data:
            logger.warning("bureau_balance表或bureau表不存在，跳过聚合")
            return main_df

        bureau = self.data["bureau"]
        bureau_balance = self.data["bureau_balance"]

        # 按SK_ID_BUREAU聚合
        bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "mean"]}

        bb_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(bb_aggregations)
        bb_agg.columns = ["_".join(col).strip() for col in bb_agg.columns.values]

        # 合并到bureau表
        bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

        # 再按SK_ID_CURR聚合
        bureau_agg = bureau.groupby("SK_ID_CURR").agg(
            {
                "MONTHS_BALANCE_min": ["min"],
                "MONTHS_BALANCE_max": ["max"],
                "MONTHS_BALANCE_mean": ["mean"],
            }
        )
        bureau_agg.columns = [
            "_".join(col).strip() for col in bureau_agg.columns.values
        ]

        # 合并到主表
        main_df = main_df.merge(bureau_agg, on="SK_ID_CURR", how="left")

        logger.info(f"bureau_balance聚合完成，添加了{bureau_agg.shape[1]}个特征")
        return main_df

    def aggregate_previous_applications(self, main_df):
        """聚合previous_application表数据"""
        logger.info("聚合previous_application表数据")

        if "previous_application" not in self.data:
            logger.warning("previous_application表不存在，跳过聚合")
            return main_df

        prev = self.data["previous_application"]

        # 数值列聚合
        num_aggregations = {
            "AMT_ANNUITY": ["min", "max", "mean"],
            "AMT_APPLICATION": ["min", "max", "mean"],
            "AMT_CREDIT": ["min", "max", "mean"],
            "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
            "AMT_GOODS_PRICE": ["min", "max", "mean"],
            "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
            "DAYS_DECISION": ["min", "max", "mean"],
            "CNT_PAYMENT": ["mean", "sum"],
        }

        # 分类列聚合
        cat_aggregations = {}
        for cat in prev.select_dtypes(include=["object"]).columns:
            if cat != "SK_ID_CURR":  # 排除ID列
                cat_aggregations[cat] = ["count"]

        prev_agg = prev.groupby("SK_ID_CURR").agg(
            {**num_aggregations, **cat_aggregations}
        )
        prev_agg.columns = ["_".join(col).strip() for col in prev_agg.columns.values]

        # 创建新特征
        prev_agg["PREV_APPLICATIONS_COUNT"] = prev.groupby("SK_ID_CURR").size()

        # 合并到主表
        main_df = main_df.merge(prev_agg, on="SK_ID_CURR", how="left")

        logger.info(f"previous_application聚合完成，添加了{prev_agg.shape[1]}个特征")
        return main_df

    def create_domain_features(self, main_df):
        """创建基于领域知识的特征"""
        logger.info("创建领域特征")

        # 债务收入比
        if (
            "AMT_CREDIT_SUM_DEBT_sum" in main_df.columns
            and "AMT_INCOME_TOTAL" in main_df.columns
        ):
            main_df["DEBT_TO_INCOME"] = (
                main_df["AMT_CREDIT_SUM_DEBT_sum"] / main_df["AMT_INCOME_TOTAL"]
            )
            main_df["DEBT_TO_INCOME"] = main_df["DEBT_TO_INCOME"].replace(
                [np.inf, -np.inf], np.nan
            )

        # 信用额度与收入比
        if "AMT_CREDIT" in main_df.columns and "AMT_INCOME_TOTAL" in main_df.columns:
            main_df["CREDIT_TO_INCOME"] = (
                main_df["AMT_CREDIT"] / main_df["AMT_INCOME_TOTAL"]
            )
            main_df["CREDIT_TO_INCOME"] = main_df["CREDIT_TO_INCOME"].replace(
                [np.inf, -np.inf], np.nan
            )

        # 年龄特征（DAYS_BIRTH是负值，表示出生至今的天数）
        if "DAYS_BIRTH" in main_df.columns:
            main_df["AGE_YEARS"] = abs(main_df["DAYS_BIRTH"]) / 365.25

        # 就业时间特征（DAYS_EMPLOYED是负值，表示就业至今的天数）
        if "DAYS_EMPLOYED" in main_df.columns:
            main_df["EMPLOYMENT_YEARS"] = abs(main_df["DAYS_EMPLOYED"]) / 365.25
            # 处理异常值（如999年的就业时间）
            main_df["EMPLOYMENT_YEARS"] = main_df["EMPLOYMENT_YEARS"].replace(
                365243.0 / 365.25, np.nan
            )

        # 就业收入比
        if (
            "EMPLOYMENT_YEARS" in main_df.columns
            and "AMT_INCOME_TOTAL" in main_df.columns
        ):
            main_df["INCOME_PER_EMPLOYMENT_YEAR"] = (
                main_df["AMT_INCOME_TOTAL"] / main_df["EMPLOYMENT_YEARS"]
            )
            main_df["INCOME_PER_EMPLOYMENT_YEAR"] = main_df[
                "INCOME_PER_EMPLOYMENT_YEAR"
            ].replace([np.inf, -np.inf], np.nan)

        logger.info("领域特征创建完成")
        return main_df

    def aggregate_pos_cash_balance(self, main_df):
        """聚合POS_CASH_balance表数据"""
        logger.info("聚合POS_CASH_balance表数据")

        if "pos_cash_balance" not in self.data:
            logger.warning("pos_cash_balance表不存在，跳过聚合")
            return main_df

        pos = self.data["pos_cash_balance"]

        # 数值列聚合
        num_aggregations = {
            "MONTHS_BALANCE": ["min", "max", "mean", "size"],
            "CNT_INSTALMENT": ["min", "max", "mean", "sum"],
            "CNT_INSTALMENT_FUTURE": ["min", "max", "mean", "sum"],
            "SK_DPD": ["min", "max", "mean", "sum"],
            "SK_DPD_DEF": ["min", "max", "mean", "sum"],
        }

        pos_agg = pos.groupby("SK_ID_CURR").agg(num_aggregations)
        pos_agg.columns = ["_".join(col).strip() for col in pos_agg.columns.values]

        # 合并到主表
        main_df = main_df.merge(pos_agg, on="SK_ID_CURR", how="left")

        logger.info(f"pos_cash_balance聚合完成，添加了{pos_agg.shape[1]}个特征")
        return main_df

    def aggregate_credit_card_balance(self, main_df):
        """聚合credit_card_balance表数据"""
        logger.info("聚合credit_card_balance表数据")

        if "credit_card_balance" not in self.data:
            logger.warning("credit_card_balance表不存在，跳过聚合")
            return main_df

        cc = self.data["credit_card_balance"]

        # 数值列聚合
        num_aggregations = {
            "MONTHS_BALANCE": ["min", "max", "mean", "size"],
            "AMT_BALANCE": ["min", "max", "mean", "sum"],
            "AMT_CREDIT_LIMIT_ACTUAL": ["min", "max", "mean"],
            "AMT_DRAWINGS_ATM_CURRENT": ["min", "max", "mean", "sum"],
            "AMT_DRAWINGS_CURRENT": ["min", "max", "mean", "sum"],
            "AMT_DRAWINGS_OTHER_CURRENT": ["min", "max", "mean", "sum"],
            "AMT_DRAWINGS_POS_CURRENT": ["min", "max", "mean", "sum"],
            "AMT_INST_MIN_REGULARITY": ["min", "max", "mean"],
            "AMT_PAYMENT_CURRENT": ["min", "max", "mean", "sum"],
            "AMT_PAYMENT_TOTAL_CURRENT": ["min", "max", "mean", "sum"],
            "AMT_RECEIVABLE_PRINCIPAL": ["min", "max", "mean", "sum"],
            "AMT_RECIVABLE": ["min", "max", "mean", "sum"],
            "AMT_TOTAL_RECEIVABLE": ["min", "max", "mean", "sum"],
            "CNT_DRAWINGS_ATM_CURRENT": ["min", "max", "mean", "sum"],
            "CNT_DRAWINGS_CURRENT": ["min", "max", "mean", "sum"],
            "CNT_DRAWINGS_OTHER_CURRENT": ["min", "max", "mean", "sum"],
            "CNT_DRAWINGS_POS_CURRENT": ["min", "max", "mean", "sum"],
            "CNT_INSTALMENT_MATURE_CUM": ["min", "max", "mean", "sum"],
            "SK_DPD": ["min", "max", "mean", "sum"],
            "SK_DPD_DEF": ["min", "max", "mean", "sum"],
        }

        cc_agg = cc.groupby("SK_ID_CURR").agg(num_aggregations)
        cc_agg.columns = ["_".join(col).strip() for col in cc_agg.columns.values]

        # 合并到主表
        main_df = main_df.merge(cc_agg, on="SK_ID_CURR", how="left")

        logger.info(f"credit_card_balance聚合完成，添加了{cc_agg.shape[1]}个特征")
        return main_df

    def aggregate_installments_payments(self, main_df):
        """聚合installments_payments表数据"""
        logger.info("聚合installments_payments表数据")

        if "installments_payments" not in self.data:
            logger.warning("installments_payments表不存在，跳过聚合")
            return main_df

        ins = self.data["installments_payments"]

        # 数值列聚合
        num_aggregations = {
            "NUM_INSTALMENT_VERSION": ["min", "max", "mean"],
            "NUM_INSTALMENT_NUMBER": ["min", "max", "mean"],
            "DAYS_INSTALMENT": ["min", "max", "mean"],
            "DAYS_ENTRY_PAYMENT": ["min", "max", "mean"],
            "AMT_INSTALMENT": ["min", "max", "mean", "sum"],
            "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        }

        ins_agg = ins.groupby("SK_ID_CURR").agg(num_aggregations)
        ins_agg.columns = ["_".join(col).strip() for col in ins_agg.columns.values]

        # 创建还款行为特征
        ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
        ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
        ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
        ins["DBD"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
        ins["DPD"] = ins["DPD"].apply(lambda x: x if x > 0 else 0)
        ins["DBD"] = ins["DBD"].apply(lambda x: x if x > 0 else 0)

        # 聚合还款行为特征
        payment_agg = ins.groupby("SK_ID_CURR").agg(
            {
                "PAYMENT_PERC": ["min", "max", "mean"],
                "PAYMENT_DIFF": ["min", "max", "mean", "sum"],
                "DPD": ["min", "max", "mean", "sum"],
                "DBD": ["min", "max", "mean", "sum"],
            }
        )
        payment_agg.columns = [
            "_".join(col).strip() for col in payment_agg.columns.values
        ]

        # 合并所有聚合特征
        ins_agg = ins_agg.merge(
            payment_agg, left_index=True, right_index=True, how="left"
        )

        # 合并到主表
        main_df = main_df.merge(ins_agg, on="SK_ID_CURR", how="left")

        logger.info(f"installments_payments聚合完成，添加了{ins_agg.shape[1]}个特征")
        return main_df

    def engineer_features(self, main_df):
        """执行所有特征工程步骤"""
        logger.info("开始特征工程")

        # 聚合所有辅助表
        main_df = self.aggregate_bureau(main_df)
        main_df = self.aggregate_bureau_balance(main_df)
        main_df = self.aggregate_previous_applications(main_df)
        main_df = self.aggregate_pos_cash_balance(main_df)
        main_df = self.aggregate_credit_card_balance(main_df)
        main_df = self.aggregate_installments_payments(main_df)

        # 创建领域特征
        main_df = self.create_domain_features(main_df)

        # 创建交互特征
        main_df = self.create_interaction_features(main_df)

        logger.info(f"特征工程完成，最终特征数量: {main_df.shape[1]}")
        return main_df

    def create_interaction_features(self, main_df):
        """创建交互特征"""
        logger.info("创建交互特征")

        # 重要特征列表（基于之前的特征重要性）- 减少特征数量以提高性能
        important_features = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "AMT_CREDIT",
            "AMT_INCOME_TOTAL",
            "AGE_YEARS",
        ]

        # 只保留存在的特征
        available_features = [f for f in important_features if f in main_df.columns]

        # 创建所有交互特征的字典，然后一次性添加
        new_features = {}

        # 创建特征交互
        for i, feat1 in enumerate(available_features):
            for feat2 in available_features[i + 1 :]:
                # 乘积交互
                new_features[f"{feat1}_X_{feat2}"] = main_df[feat1] * main_df[feat2]

                # 比值交互（避免除零）
                new_features[f"{feat1}_DIV_{feat2}"] = main_df[feat1] / (
                    main_df[feat2] + 1e-8
                )

                # 差值交互
                new_features[f"{feat1}_MINUS_{feat2}"] = main_df[feat1] - main_df[feat2]

        # EXT_SOURCE特征的特殊组合
        if all(f"EXT_SOURCE_{i}" in main_df.columns for i in [1, 2, 3]):
            new_features["EXT_SOURCE_MEAN"] = (
                main_df["EXT_SOURCE_1"]
                + main_df["EXT_SOURCE_2"]
                + main_df["EXT_SOURCE_3"]
            ) / 3
            new_features["EXT_SOURCE_STD"] = main_df[
                ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
            ].std(axis=1)
            new_features["EXT_SOURCE_PROD"] = (
                main_df["EXT_SOURCE_1"]
                * main_df["EXT_SOURCE_2"]
                * main_df["EXT_SOURCE_3"]
            )

        # 将所有新特征转换为DataFrame并一次性连接
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=main_df.index)
            # 处理无穷大值
            new_features_df = new_features_df.replace([np.inf, -np.inf], np.nan)
            # 一次性连接所有新特征
            main_df = pd.concat([main_df, new_features_df], axis=1)

        logger.info("交互特征创建完成")
        return main_df
