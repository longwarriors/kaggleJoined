"""
数据验证模块

提供数据完整性、一致性和质量验证功能。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from ..core.base import BaseProcessor


class DataValidator(BaseProcessor):
    """
    数据验证器
    
    验证数据的完整性、一致性和质量
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化数据验证器
        
        Args:
            config: 验证配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.validation_results = {}
        
    def fit(self, data: Dict[str, pd.DataFrame], **kwargs) -> 'DataValidator':
        """
        拟合数据验证器
        
        Args:
            data: 数据字典
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.data = data
        self.is_fitted = True
        return self
    
    def transform(self, data: Dict[str, pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        执行数据验证
        
        Args:
            data: 数据字典，如果不提供则使用fit时的数据
            **kwargs: 其他参数
            
        Returns:
            验证结果字典
        """
        if data is not None:
            self.data = data
        
        return self.validate_all()
    
    def validate_all(self) -> Dict[str, Any]:
        """
        执行完整的数据验证
        
        Returns:
            完整的验证结果
        """
        self._log_info("开始执行数据验证")
        
        results = {}
        
        # 1. 基本完整性验证
        results['integrity'] = self.validate_integrity()
        
        # 2. 数据类型验证
        results['dtypes'] = self.validate_dtypes()
        
        # 3. 数据范围验证
        results['ranges'] = self.validate_ranges()
        
        # 4. 业务规则验证
        results['business_rules'] = self.validate_business_rules()
        
        # 5. 数据一致性验证
        results['consistency'] = self.validate_consistency()
        
        # 6. 数据质量评分
        results['quality_score'] = self.calculate_quality_score(results)
        
        self.validation_results = results
        self._log_info("数据验证完成")
        
        return results
    
    def validate_integrity(self) -> Dict[str, Any]:
        """
        验证数据完整性
        
        Returns:
            完整性验证结果
        """
        integrity_results = {}
        validation_config = self.config.get('validation', {})
        
        for name, df in self.data.items():
            result = {
                'passed': True,
                'issues': [],
                'warnings': []
            }
            
            # 检查必需列
            required_cols = validation_config.get('required_columns', {}).get(name, [])
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                result['passed'] = False
                result['issues'].append(f"缺少必需列: {missing_cols}")
            
            # 检查数据形状
            expected_shapes = validation_config.get('expected_shapes', {})
            if name in expected_shapes:
                min_rows, max_rows = expected_shapes[name]
                actual_rows = len(df)
                if not (min_rows <= actual_rows <= max_rows):
                    result['warnings'].append(
                        f"数据行数 {actual_rows} 超出预期范围 [{min_rows}, {max_rows}]"
                    )
            
            # 检查重复行
            if validation_config.get('check_duplicates', True):
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    result['warnings'].append(f"发现 {duplicates} 行重复数据")
            
            # 检查空DataFrame
            if df.empty:
                result['passed'] = False
                result['issues'].append("数据为空")
            
            integrity_results[name] = result
        
        return integrity_results
    
    def validate_dtypes(self) -> Dict[str, Any]:
        """
        验证数据类型
        
        Returns:
            数据类型验证结果
        """
        dtype_results = {}
        
        for name, df in self.data.items():
            result = {
                'passed': True,
                'issues': [],
                'warnings': [],
                'type_summary': df.dtypes.value_counts().to_dict()
            }
            
            # 检查混合类型列
            for col in df.columns:
                if df[col].dtype == 'object':
                    # 检查是否应该是数值类型
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # 尝试转换为数值
                        try:
                            pd.to_numeric(non_null_values.head(100))
                            result['warnings'].append(
                                f"列 {col} 可能应该是数值类型"
                            )
                        except:
                            pass
            
            # 检查日期列
            date_like_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_like_cols:
                if df[col].dtype == 'object':
                    result['warnings'].append(
                        f"列 {col} 可能应该是日期类型"
                    )
            
            dtype_results[name] = result
        
        return dtype_results
    
    def validate_ranges(self) -> Dict[str, Any]:
        """
        验证数据范围
        
        Returns:
            数据范围验证结果
        """
        range_results = {}
        
        for name, df in self.data.items():
            result = {
                'passed': True,
                'issues': [],
                'warnings': []
            }
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                # 检查异常值
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                if len(outliers) > len(col_data) * 0.05:  # 超过5%的异常值
                    result['warnings'].append(
                        f"列 {col} 有大量异常值: {len(outliers)} ({len(outliers)/len(col_data)*100:.1f}%)"
                    )
                
                # 检查负值（对于某些业务字段不应该有负值）
                negative_cols = ['AMT_', 'CNT_', 'DAYS_BIRTH']  # 这些前缀的列通常不应该有负值（除了DAYS_BIRTH）
                if any(prefix in col for prefix in negative_cols[:-1]) and (col_data < 0).any():
                    result['warnings'].append(f"列 {col} 包含负值")
                
                # 检查无穷大值
                if np.isinf(col_data).any():
                    result['issues'].append(f"列 {col} 包含无穷大值")
                    result['passed'] = False
            
            range_results[name] = result
        
        return range_results
    
    def validate_business_rules(self) -> Dict[str, Any]:
        """
        验证业务规则
        
        Returns:
            业务规则验证结果
        """
        business_results = {}
        
        # 主要针对application_train和application_test
        for name in ['application_train', 'application_test']:
            if name not in self.data:
                continue
                
            df = self.data[name]
            result = {
                'passed': True,
                'issues': [],
                'warnings': []
            }
            
            # 检查年龄合理性（DAYS_BIRTH应该是负值，表示出生至今的天数）
            if 'DAYS_BIRTH' in df.columns:
                age_years = -df['DAYS_BIRTH'] / 365.25
                if (age_years < 18).any() or (age_years > 100).any():
                    result['warnings'].append("发现不合理的年龄值")
            
            # 检查收入合理性
            if 'AMT_INCOME_TOTAL' in df.columns:
                income = df['AMT_INCOME_TOTAL']
                if (income <= 0).any():
                    result['warnings'].append("发现零或负收入")
                if (income > 1e8).any():  # 收入超过1亿
                    result['warnings'].append("发现异常高收入")
            
            # 检查信贷金额合理性
            if 'AMT_CREDIT' in df.columns:
                credit = df['AMT_CREDIT']
                if (credit <= 0).any():
                    result['warnings'].append("发现零或负信贷金额")
            
            # 检查年金与信贷金额的关系
            if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
                annuity_credit_ratio = df['AMT_ANNUITY'] / df['AMT_CREDIT']
                if (annuity_credit_ratio > 1).any():
                    result['warnings'].append("发现年金大于信贷金额的情况")
            
            # 检查目标变量（仅对训练集）
            if name == 'application_train' and 'TARGET' in df.columns:
                target = df['TARGET']
                unique_values = target.unique()
                if not set(unique_values).issubset({0, 1}):
                    result['issues'].append("目标变量包含非0/1值")
                    result['passed'] = False
            
            business_results[name] = result
        
        return business_results
    
    def validate_consistency(self) -> Dict[str, Any]:
        """
        验证数据一致性
        
        Returns:
            一致性验证结果
        """
        consistency_results = {}
        
        # 检查训练集和测试集的一致性
        if 'application_train' in self.data and 'application_test' in self.data:
            train_df = self.data['application_train']
            test_df = self.data['application_test']
            
            result = {
                'passed': True,
                'issues': [],
                'warnings': []
            }
            
            # 检查列的一致性
            train_cols = set(train_df.columns) - {'TARGET'}  # 排除目标列
            test_cols = set(test_df.columns)
            
            missing_in_test = train_cols - test_cols
            extra_in_test = test_cols - train_cols
            
            if missing_in_test:
                result['issues'].append(f"测试集缺少列: {missing_in_test}")
                result['passed'] = False
            
            if extra_in_test:
                result['warnings'].append(f"测试集多出列: {extra_in_test}")
            
            # 检查共同列的数据类型一致性
            common_cols = train_cols & test_cols
            for col in common_cols:
                if train_df[col].dtype != test_df[col].dtype:
                    result['warnings'].append(
                        f"列 {col} 数据类型不一致: 训练集 {train_df[col].dtype}, 测试集 {test_df[col].dtype}"
                    )
            
            consistency_results['train_test'] = result
        
        return consistency_results
    
    def calculate_quality_score(self, validation_results: Dict[str, Any]) -> Dict[str, float]:
        """
        计算数据质量评分
        
        Args:
            validation_results: 验证结果
            
        Returns:
            质量评分字典
        """
        quality_scores = {}
        
        for name in self.data.keys():
            score = 100.0  # 初始分数
            
            # 完整性扣分
            if name in validation_results.get('integrity', {}):
                integrity = validation_results['integrity'][name]
                if not integrity['passed']:
                    score -= 30
                score -= len(integrity['warnings']) * 5
            
            # 数据类型扣分
            if name in validation_results.get('dtypes', {}):
                dtypes = validation_results['dtypes'][name]
                score -= len(dtypes['warnings']) * 3
            
            # 数据范围扣分
            if name in validation_results.get('ranges', {}):
                ranges = validation_results['ranges'][name]
                if not ranges['passed']:
                    score -= 20
                score -= len(ranges['warnings']) * 2
            
            # 业务规则扣分
            if name in validation_results.get('business_rules', {}):
                business = validation_results['business_rules'][name]
                if not business['passed']:
                    score -= 25
                score -= len(business['warnings']) * 3
            
            quality_scores[name] = max(0, score)
        
        return quality_scores
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        获取验证摘要
        
        Returns:
            验证摘要字典
        """
        if not self.validation_results:
            return {}
        
        summary = {
            'overall_passed': True,
            'total_issues': 0,
            'total_warnings': 0,
            'datasets_summary': {}
        }
        
        for dataset_name in self.data.keys():
            dataset_summary = {
                'passed': True,
                'issues': 0,
                'warnings': 0,
                'quality_score': self.validation_results.get('quality_score', {}).get(dataset_name, 0)
            }
            
            # 统计各类验证的问题
            for validation_type in ['integrity', 'dtypes', 'ranges', 'business_rules']:
                if validation_type in self.validation_results:
                    type_results = self.validation_results[validation_type]
                    if dataset_name in type_results:
                        result = type_results[dataset_name]
                        if not result.get('passed', True):
                            dataset_summary['passed'] = False
                        dataset_summary['issues'] += len(result.get('issues', []))
                        dataset_summary['warnings'] += len(result.get('warnings', []))
            
            summary['datasets_summary'][dataset_name] = dataset_summary
            summary['total_issues'] += dataset_summary['issues']
            summary['total_warnings'] += dataset_summary['warnings']
            
            if not dataset_summary['passed']:
                summary['overall_passed'] = False
        
        return summary
