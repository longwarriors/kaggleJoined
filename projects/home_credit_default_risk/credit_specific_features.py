"""
Home Credit专用特征工程模块

基于Home Credit Default Risk竞赛的特定数据结构和业务逻辑，
创建更有针对性的特征，专注于信用风险评估的关键指标。

主要特征类别：
1. 信贷历史聚合特征
2. 还款行为特征  
3. 申请行为特征
4. 财务状况特征
5. 领域知识特征

作者：Augment Agent
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CreditSpecificFeatureEngineer:
    """Home Credit专用特征工程器"""
    
    def __init__(self, data_dict):
        """
        初始化特征工程器
        
        参数:
            data_dict (dict): 包含所有数据集的字典
        """
        self.data = data_dict
    
    def create_bureau_features(self, main_df):
        """创建信贷局相关特征"""
        logger.info("创建信贷局特征")
        
        if "bureau" not in self.data:
            logger.warning("bureau数据不存在，跳过相关特征")
            return main_df
        
        bureau = self.data["bureau"].copy()
        
        # 基础聚合特征
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({
            'DAYS_CREDIT': ['count', 'mean', 'min', 'max', 'std'],
            'CREDIT_DAY_OVERDUE': ['mean', 'max', 'sum'],
            'DAYS_CREDIT_ENDDATE': ['mean', 'min', 'max'],
            'DAYS_ENDDATE_FACT': ['mean', 'min', 'max'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max', 'sum'],
            'CNT_CREDIT_PROLONG': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM': ['sum', 'mean', 'max', 'std'],
            'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max', 'std'],
            'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean', 'max', 'std'],
            'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
            'CREDIT_TYPE': ['nunique'],
            'CREDIT_ACTIVE': ['nunique']
        })
        
        # 扁平化列名
        bureau_agg.columns = ['BUREAU_' + '_'.join(col).strip() for col in bureau_agg.columns.values]
        
        # 创建比率特征
        bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = (
            bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_sum'] / 
            (bureau_agg['BUREAU_AMT_CREDIT_SUM_sum'] + 1e-8)
        )
        
        bureau_agg['BUREAU_OVERDUE_CREDIT_RATIO'] = (
            bureau_agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_sum'] / 
            (bureau_agg['BUREAU_AMT_CREDIT_SUM_sum'] + 1e-8)
        )
        
        # 活跃信贷特征
        active_bureau = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
        if not active_bureau.empty:
            active_agg = active_bureau.groupby('SK_ID_CURR').agg({
                'AMT_CREDIT_SUM': ['sum', 'mean'],
                'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
                'CREDIT_DAY_OVERDUE': ['mean', 'max']
            })
            active_agg.columns = ['BUREAU_ACTIVE_' + '_'.join(col).strip() for col in active_agg.columns.values]
            bureau_agg = bureau_agg.join(active_agg, how='left')
        
        # 合并到主数据
        main_df = main_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        
        logger.info(f"创建了{len(bureau_agg.columns)}个信贷局特征")
        return main_df
    
    def create_previous_application_features(self, main_df):
        """创建历史申请特征"""
        logger.info("创建历史申请特征")
        
        if "previous_application" not in self.data:
            logger.warning("previous_application数据不存在，跳过相关特征")
            return main_df
        
        prev = self.data["previous_application"].copy()
        
        # 基础聚合特征
        prev_agg = prev.groupby('SK_ID_CURR').agg({
            'AMT_ANNUITY': ['mean', 'max', 'min', 'std'],
            'AMT_APPLICATION': ['mean', 'max', 'min', 'std'],
            'AMT_CREDIT': ['mean', 'max', 'min', 'std'],
            'AMT_DOWN_PAYMENT': ['mean', 'max', 'min', 'std'],
            'AMT_GOODS_PRICE': ['mean', 'max', 'min', 'std'],
            'HOUR_APPR_PROCESS_START': ['mean', 'max', 'min', 'std'],
            'RATE_DOWN_PAYMENT': ['mean', 'max', 'min', 'std'],
            'DAYS_DECISION': ['mean', 'max', 'min', 'std'],
            'CNT_PAYMENT': ['mean', 'max', 'min', 'std'],
            'NAME_CONTRACT_STATUS': ['count'],
            'NAME_PAYMENT_TYPE': ['nunique'],
            'CODE_REJECT_REASON': ['nunique']
        })
        
        # 扁平化列名
        prev_agg.columns = ['PREV_' + '_'.join(col).strip() for col in prev_agg.columns.values]
        
        # 申请状态特征
        status_counts = prev.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].value_counts().unstack(fill_value=0)
        status_counts.columns = ['PREV_STATUS_' + col for col in status_counts.columns]
        prev_agg = prev_agg.join(status_counts, how='left')
        
        # 批准率特征
        if 'PREV_STATUS_Approved' in prev_agg.columns:
            prev_agg['PREV_APPROVAL_RATE'] = (
                prev_agg['PREV_STATUS_Approved'] / 
                (prev_agg['PREV_NAME_CONTRACT_STATUS_count'] + 1e-8)
            )
        
        # 最近申请特征
        recent_prev = prev.sort_values('DAYS_DECISION').groupby('SK_ID_CURR').tail(1)
        recent_features = recent_prev.set_index('SK_ID_CURR')[
            ['AMT_CREDIT', 'AMT_ANNUITY', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION']
        ]
        recent_features.columns = ['PREV_RECENT_' + col for col in recent_features.columns]
        prev_agg = prev_agg.join(recent_features, how='left')
        
        # 合并到主数据
        main_df = main_df.merge(prev_agg, on='SK_ID_CURR', how='left')
        
        logger.info(f"创建了{len(prev_agg.columns)}个历史申请特征")
        return main_df
    
    def create_credit_card_features(self, main_df):
        """创建信用卡特征"""
        logger.info("创建信用卡特征")
        
        if "credit_card_balance" not in self.data:
            logger.warning("credit_card_balance数据不存在，跳过相关特征")
            return main_df
        
        cc = self.data["credit_card_balance"].copy()
        
        # 计算使用率
        cc['BALANCE_LIMIT_RATIO'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-8)
        cc['PAYMENT_MIN_RATIO'] = cc['AMT_PAYMENT_CURRENT'] / (cc['AMT_INST_MIN_REGULARITY'] + 1e-8)
        
        # 聚合特征
        cc_agg = cc.groupby('SK_ID_CURR').agg({
            'MONTHS_BALANCE': ['count', 'mean', 'min', 'max'],
            'AMT_BALANCE': ['mean', 'max', 'min', 'std'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max', 'min'],
            'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['mean', 'max', 'sum'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['mean', 'max', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['mean', 'max', 'sum'],
            'AMT_PAYMENT_CURRENT': ['mean', 'max', 'sum'],
            'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'sum'],
            'CNT_DRAWINGS_CURRENT': ['mean', 'max', 'sum'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['mean', 'max', 'sum'],
            'CNT_DRAWINGS_POS_CURRENT': ['mean', 'max', 'sum'],
            'SK_DPD': ['mean', 'max', 'sum'],
            'SK_DPD_DEF': ['mean', 'max', 'sum'],
            'BALANCE_LIMIT_RATIO': ['mean', 'max', 'min'],
            'PAYMENT_MIN_RATIO': ['mean', 'max', 'min']
        })
        
        # 扁平化列名
        cc_agg.columns = ['CC_' + '_'.join(col).strip() for col in cc_agg.columns.values]
        
        # 合并到主数据
        main_df = main_df.merge(cc_agg, on='SK_ID_CURR', how='left')
        
        logger.info(f"创建了{len(cc_agg.columns)}个信用卡特征")
        return main_df
    
    def create_domain_knowledge_features(self, main_df):
        """创建基于领域知识的特征"""
        logger.info("创建领域知识特征")
        
        # 收入相关特征
        if 'AMT_INCOME_TOTAL' in main_df.columns:
            # 收入稳定性（基于收入类型）
            income_stability = {
                'Working': 1.0,
                'Commercial associate': 0.8,
                'Pensioner': 0.9,
                'State servant': 0.95,
                'Student': 0.3,
                'Businessman': 0.6,
                'Maternity leave': 0.4,
                'Unemployed': 0.1
            }
            
            if 'NAME_INCOME_TYPE' in main_df.columns:
                main_df['INCOME_STABILITY_SCORE'] = main_df['NAME_INCOME_TYPE'].map(income_stability).fillna(0.5)
                main_df['ADJUSTED_INCOME'] = main_df['AMT_INCOME_TOTAL'] * main_df['INCOME_STABILITY_SCORE']
        
        # 信贷负担特征
        if all(col in main_df.columns for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY']):
            main_df['CREDIT_INCOME_RATIO'] = main_df['AMT_CREDIT'] / (main_df['AMT_INCOME_TOTAL'] + 1e-8)
            main_df['ANNUITY_INCOME_RATIO'] = main_df['AMT_ANNUITY'] / (main_df['AMT_INCOME_TOTAL'] + 1e-8)
            main_df['CREDIT_ANNUITY_RATIO'] = main_df['AMT_CREDIT'] / (main_df['AMT_ANNUITY'] + 1e-8)
        
        # 年龄相关特征
        if 'DAYS_BIRTH' in main_df.columns:
            main_df['AGE_YEARS'] = -main_df['DAYS_BIRTH'] / 365.25
            main_df['AGE_GROUP'] = pd.cut(main_df['AGE_YEARS'], 
                                        bins=[0, 25, 35, 45, 55, 65, 100], 
                                        labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder', 'Old'])
        
        # 就业稳定性
        if 'DAYS_EMPLOYED' in main_df.columns:
            main_df['EMPLOYMENT_YEARS'] = -main_df['DAYS_EMPLOYED'] / 365.25
            main_df['EMPLOYMENT_YEARS'] = main_df['EMPLOYMENT_YEARS'].clip(lower=0, upper=50)  # 处理异常值
        
        # EXT_SOURCE组合特征
        ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        available_ext = [col for col in ext_sources if col in main_df.columns]
        
        if len(available_ext) >= 2:
            # 创建EXT_SOURCE的各种组合
            main_df['EXT_SOURCE_MEAN'] = main_df[available_ext].mean(axis=1)
            main_df['EXT_SOURCE_STD'] = main_df[available_ext].std(axis=1)
            main_df['EXT_SOURCE_MAX'] = main_df[available_ext].max(axis=1)
            main_df['EXT_SOURCE_MIN'] = main_df[available_ext].min(axis=1)
            
            # 两两乘积
            for i in range(len(available_ext)):
                for j in range(i+1, len(available_ext)):
                    main_df[f'{available_ext[i]}_X_{available_ext[j]}'] = (
                        main_df[available_ext[i]] * main_df[available_ext[j]]
                    )
        
        logger.info("领域知识特征创建完成")
        return main_df
    
    def engineer_all_features(self, main_df):
        """执行所有专用特征工程"""
        logger.info("开始Home Credit专用特征工程")
        
        original_features = main_df.shape[1]
        
        # 1. 信贷局特征
        main_df = self.create_bureau_features(main_df)
        
        # 2. 历史申请特征
        main_df = self.create_previous_application_features(main_df)
        
        # 3. 信用卡特征
        main_df = self.create_credit_card_features(main_df)
        
        # 4. 领域知识特征
        main_df = self.create_domain_knowledge_features(main_df)
        
        new_features = main_df.shape[1] - original_features
        logger.info(f"Home Credit专用特征工程完成，新增{new_features}个特征，总特征数: {main_df.shape[1]}")
        
        return main_df
