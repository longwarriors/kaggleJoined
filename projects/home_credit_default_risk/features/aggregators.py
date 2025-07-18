"""
特征聚合模块

实现从辅助表聚合特征的功能，包括数值聚合、类别聚合等。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseProcessor


class FeatureAggregators(BaseProcessor):
    """
    特征聚合器
    
    从辅助表聚合特征到主表
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化特征聚合器
        
        Args:
            config: 聚合配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.aggregation_stats = {}
        
    def fit(self, data: Dict[str, pd.DataFrame], **kwargs) -> 'FeatureAggregators':
        """
        拟合特征聚合器
        
        Args:
            data: 数据字典
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.data = data
        self.is_fitted = True
        return self
    
    def transform(self, data: Dict[str, pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """
        执行特征聚合
        
        Args:
            data: 数据字典，如果不提供则使用fit时的数据
            **kwargs: 其他参数
            
        Returns:
            聚合后的特征DataFrame
        """
        if data is not None:
            data_to_aggregate = data
        else:
            data_to_aggregate = self.data
        
        return self.aggregate_all_tables(data_to_aggregate)
    
    def aggregate_all_tables(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        聚合所有辅助表
        
        Args:
            data: 数据字典
            
        Returns:
            聚合特征DataFrame
        """
        self._log_info("开始聚合辅助表特征")
        
        aggregation_config = self.config.get('aggregation_features', {})
        
        if not aggregation_config.get('enable', True):
            # 返回空的聚合结果，只包含SK_ID_CURR
            main_df = data.get('application_train') or data.get('application_test')
            if main_df is not None:
                return pd.DataFrame({'SK_ID_CURR': main_df['SK_ID_CURR']})
            else:
                return pd.DataFrame()
        
        tables_config = aggregation_config.get('tables', {})
        
        # 初始化结果DataFrame
        main_df = data.get('application_train') or data.get('application_test')
        if main_df is None:
            raise ValueError("未找到主表数据")
        
        result_df = pd.DataFrame({'SK_ID_CURR': main_df['SK_ID_CURR']})
        
        # 逐个处理辅助表
        for table_name, table_config in tables_config.items():
            if table_name in data:
                self._log_info(f"聚合表: {table_name}")
                aggregated_features = self._aggregate_single_table(
                    data[table_name], table_name, table_config
                )
                
                if aggregated_features is not None and not aggregated_features.empty:
                    result_df = result_df.merge(
                        aggregated_features, 
                        on='SK_ID_CURR', 
                        how='left'
                    )
                    
                    self._log_info(f"{table_name} 聚合完成，添加了 {aggregated_features.shape[1]-1} 个特征")
            else:
                self._log_warning(f"表 {table_name} 不存在，跳过聚合")
        
        self._log_info(f"所有表聚合完成，总特征数: {result_df.shape[1]-1}")
        return result_df
    
    def _aggregate_single_table(self, df: pd.DataFrame, table_name: str, 
                               table_config: Dict) -> Optional[pd.DataFrame]:
        """
        聚合单个表
        
        Args:
            df: 表数据
            table_name: 表名
            table_config: 表配置
            
        Returns:
            聚合后的特征DataFrame
        """
        group_key = table_config.get('group_key', 'SK_ID_CURR')
        
        if group_key not in df.columns:
            self._log_warning(f"表 {table_name} 缺少分组键 {group_key}")
            return None
        
        # 数值列聚合
        numerical_aggs = table_config.get('numerical_aggs', {})
        categorical_aggs = table_config.get('categorical_aggs', {})
        
        all_aggs = {}
        
        # 处理数值聚合
        for col, agg_funcs in numerical_aggs.items():
            if col in df.columns:
                # 确保列是数值类型
                if df[col].dtype in ['object', 'category']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        self._log_warning(f"无法将列 {col} 转换为数值类型")
                        continue
                
                all_aggs[col] = agg_funcs
        
        # 处理类别聚合
        for col, agg_funcs in categorical_aggs.items():
            if col in df.columns:
                all_aggs[col] = agg_funcs
        
        if not all_aggs:
            self._log_warning(f"表 {table_name} 没有可聚合的列")
            return None
        
        try:
            # 执行聚合
            agg_result = df.groupby(group_key).agg(all_aggs)
            
            # 扁平化列名
            if isinstance(agg_result.columns, pd.MultiIndex):
                agg_result.columns = [
                    f'{table_name}_{col}_{agg}' 
                    for col, agg in agg_result.columns
                ]
            else:
                agg_result.columns = [
                    f'{table_name}_{col}' 
                    for col in agg_result.columns
                ]
            
            # 重置索引
            agg_result = agg_result.reset_index()
            
            # 创建额外的聚合特征
            agg_result = self._create_additional_features(agg_result, df, table_name, table_config)
            
            return agg_result
            
        except Exception as e:
            self._log_error(f"聚合表 {table_name} 失败: {e}")
            return None
    
    def _create_additional_features(self, agg_df: pd.DataFrame, original_df: pd.DataFrame,
                                   table_name: str, table_config: Dict) -> pd.DataFrame:
        """
        创建额外的聚合特征
        
        Args:
            agg_df: 已聚合的DataFrame
            original_df: 原始表数据
            table_name: 表名
            table_config: 表配置
            
        Returns:
            添加额外特征后的DataFrame
        """
        group_key = table_config.get('group_key', 'SK_ID_CURR')
        
        # 根据表类型创建特定特征
        if table_name == 'bureau':
            agg_df = self._create_bureau_features(agg_df, original_df, group_key)
        elif table_name == 'previous_application':
            agg_df = self._create_previous_application_features(agg_df, original_df, group_key)
        elif table_name == 'credit_card_balance':
            agg_df = self._create_credit_card_features(agg_df, original_df, group_key)
        elif table_name == 'installments_payments':
            agg_df = self._create_installments_features(agg_df, original_df, group_key)
        
        return agg_df
    
    def _create_bureau_features(self, agg_df: pd.DataFrame, bureau_df: pd.DataFrame, 
                               group_key: str) -> pd.DataFrame:
        """创建bureau表的额外特征"""
        try:
            # 活跃信贷数量
            if 'CREDIT_ACTIVE' in bureau_df.columns:
                active_credits = bureau_df[bureau_df['CREDIT_ACTIVE'] == 'Active'].groupby(group_key).size()
                agg_df = agg_df.merge(
                    active_credits.to_frame('bureau_active_credits_count').reset_index(),
                    on=group_key, how='left'
                )
                agg_df['bureau_active_credits_count'] = agg_df['bureau_active_credits_count'].fillna(0)
            
            # 逾期信贷数量
            if 'CREDIT_DAY_OVERDUE' in bureau_df.columns:
                overdue_credits = bureau_df[bureau_df['CREDIT_DAY_OVERDUE'] > 0].groupby(group_key).size()
                agg_df = agg_df.merge(
                    overdue_credits.to_frame('bureau_overdue_credits_count').reset_index(),
                    on=group_key, how='left'
                )
                agg_df['bureau_overdue_credits_count'] = agg_df['bureau_overdue_credits_count'].fillna(0)
            
            # 债务比例
            debt_col = 'bureau_AMT_CREDIT_SUM_DEBT_sum'
            credit_col = 'bureau_AMT_CREDIT_SUM_sum'
            if debt_col in agg_df.columns and credit_col in agg_df.columns:
                agg_df['bureau_debt_credit_ratio'] = (
                    agg_df[debt_col] / (agg_df[credit_col] + 1e-8)
                )
            
        except Exception as e:
            self._log_warning(f"创建bureau额外特征失败: {e}")
        
        return agg_df
    
    def _create_previous_application_features(self, agg_df: pd.DataFrame, prev_df: pd.DataFrame,
                                            group_key: str) -> pd.DataFrame:
        """创建previous_application表的额外特征"""
        try:
            # 批准率
            if 'NAME_CONTRACT_STATUS' in prev_df.columns:
                approval_stats = prev_df.groupby(group_key)['NAME_CONTRACT_STATUS'].apply(
                    lambda x: (x == 'Approved').sum() / len(x) if len(x) > 0 else 0
                )
                agg_df = agg_df.merge(
                    approval_stats.to_frame('prev_approval_rate').reset_index(),
                    on=group_key, how='left'
                )
                agg_df['prev_approval_rate'] = agg_df['prev_approval_rate'].fillna(0)
            
            # 拒绝率
            if 'NAME_CONTRACT_STATUS' in prev_df.columns:
                rejection_stats = prev_df.groupby(group_key)['NAME_CONTRACT_STATUS'].apply(
                    lambda x: (x == 'Refused').sum() / len(x) if len(x) > 0 else 0
                )
                agg_df = agg_df.merge(
                    rejection_stats.to_frame('prev_rejection_rate').reset_index(),
                    on=group_key, how='left'
                )
                agg_df['prev_rejection_rate'] = agg_df['prev_rejection_rate'].fillna(0)
            
            # 最近申请的天数
            if 'DAYS_DECISION' in prev_df.columns:
                recent_application = prev_df.groupby(group_key)['DAYS_DECISION'].max()
                agg_df = agg_df.merge(
                    recent_application.to_frame('prev_most_recent_application_days').reset_index(),
                    on=group_key, how='left'
                )
            
        except Exception as e:
            self._log_warning(f"创建previous_application额外特征失败: {e}")
        
        return agg_df
    
    def _create_credit_card_features(self, agg_df: pd.DataFrame, cc_df: pd.DataFrame,
                                   group_key: str) -> pd.DataFrame:
        """创建credit_card_balance表的额外特征"""
        try:
            # 信用卡使用率
            if 'AMT_BALANCE' in cc_df.columns and 'AMT_CREDIT_LIMIT_ACTUAL' in cc_df.columns:
                cc_df['utilization_ratio'] = cc_df['AMT_BALANCE'] / (cc_df['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-8)
                utilization_stats = cc_df.groupby(group_key)['utilization_ratio'].agg(['mean', 'max'])
                utilization_stats.columns = ['cc_utilization_mean', 'cc_utilization_max']
                agg_df = agg_df.merge(utilization_stats.reset_index(), on=group_key, how='left')
            
            # 逾期次数
            if 'SK_DPD' in cc_df.columns:
                dpd_count = cc_df[cc_df['SK_DPD'] > 0].groupby(group_key).size()
                agg_df = agg_df.merge(
                    dpd_count.to_frame('cc_dpd_count').reset_index(),
                    on=group_key, how='left'
                )
                agg_df['cc_dpd_count'] = agg_df['cc_dpd_count'].fillna(0)
            
        except Exception as e:
            self._log_warning(f"创建credit_card额外特征失败: {e}")
        
        return agg_df
    
    def _create_installments_features(self, agg_df: pd.DataFrame, inst_df: pd.DataFrame,
                                    group_key: str) -> pd.DataFrame:
        """创建installments_payments表的额外特征"""
        try:
            # 还款表现
            if 'AMT_INSTALMENT' in inst_df.columns and 'AMT_PAYMENT' in inst_df.columns:
                inst_df['payment_ratio'] = inst_df['AMT_PAYMENT'] / (inst_df['AMT_INSTALMENT'] + 1e-8)
                payment_stats = inst_df.groupby(group_key)['payment_ratio'].agg(['mean', 'std'])
                payment_stats.columns = ['inst_payment_ratio_mean', 'inst_payment_ratio_std']
                agg_df = agg_df.merge(payment_stats.reset_index(), on=group_key, how='left')
            
            # 逾期天数
            if 'DAYS_ENTRY_PAYMENT' in inst_df.columns and 'DAYS_INSTALMENT' in inst_df.columns:
                inst_df['days_past_due'] = inst_df['DAYS_ENTRY_PAYMENT'] - inst_df['DAYS_INSTALMENT']
                inst_df['days_past_due'] = inst_df['days_past_due'].clip(lower=0)
                
                dpd_stats = inst_df.groupby(group_key)['days_past_due'].agg(['mean', 'max', 'sum'])
                dpd_stats.columns = ['inst_dpd_mean', 'inst_dpd_max', 'inst_dpd_sum']
                agg_df = agg_df.merge(dpd_stats.reset_index(), on=group_key, how='left')
            
        except Exception as e:
            self._log_warning(f"创建installments额外特征失败: {e}")
        
        return agg_df
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        return self.aggregation_stats.copy()
