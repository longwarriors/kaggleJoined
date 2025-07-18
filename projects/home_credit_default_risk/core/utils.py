"""
通用工具函数模块

提供项目中常用的工具函数和辅助功能。

作者：Augment Agent
"""

import os
import pickle
import joblib
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
from datetime import datetime
import gc
import psutil


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    减少DataFrame的内存使用量
    
    Args:
        df: 输入DataFrame
        verbose: 是否打印详细信息
        
    Returns:
        优化后的DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'内存使用量从 {start_mem:.2f} MB 减少到 {end_mem:.2f} MB '
              f'(减少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    
    return df


def get_memory_usage() -> Dict[str, float]:
    """
    获取当前内存使用情况
    
    Returns:
        内存使用信息字典
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
        'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
        'percent': process.memory_percent()        # 内存使用百分比
    }


def clean_memory():
    """清理内存"""
    gc.collect()


def save_object(obj: Any, filepath: str, method: str = 'joblib'):
    """
    保存对象到文件
    
    Args:
        obj: 要保存的对象
        filepath: 文件路径
        method: 保存方法 ('joblib', 'pickle')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if method == 'joblib':
        joblib.dump(obj, filepath)
    elif method == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    else:
        raise ValueError(f"不支持的保存方法: {method}")


def load_object(filepath: str, method: str = 'joblib') -> Any:
    """
    从文件加载对象
    
    Args:
        filepath: 文件路径
        method: 加载方法 ('joblib', 'pickle')
        
    Returns:
        加载的对象
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    if method == 'joblib':
        return joblib.load(filepath)
    elif method == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"不支持的加载方法: {method}")


def save_json(data: Dict, filepath: str, indent: int = 2):
    """
    保存字典到JSON文件
    
    Args:
        data: 要保存的字典
        filepath: 文件路径
        indent: 缩进空格数
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Dict:
    """
    从JSON文件加载字典
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的字典
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_directory(path: str, exist_ok: bool = True):
    """
    创建目录
    
    Args:
        path: 目录路径
        exist_ok: 如果目录已存在是否报错
    """
    Path(path).mkdir(parents=True, exist_ok=exist_ok)


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    获取当前时间戳
    
    Args:
        format_str: 时间格式字符串
        
    Returns:
        格式化的时间戳
    """
    return datetime.now().strftime(format_str)


def safe_divide(numerator: Union[float, np.ndarray], 
               denominator: Union[float, np.ndarray], 
               fill_value: float = 0.0) -> Union[float, np.ndarray]:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        fill_value: 除零时的填充值
        
    Returns:
        除法结果
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.divide(numerator, denominator)
        
    if isinstance(result, np.ndarray):
        result = np.where(np.isfinite(result), result, fill_value)
    else:
        if not np.isfinite(result):
            result = fill_value
            
    return result


def clip_outliers(data: pd.Series, method: str = 'iqr', 
                 factor: float = 1.5) -> pd.Series:
    """
    裁剪异常值
    
    Args:
        data: 输入数据
        method: 异常值检测方法 ('iqr', 'zscore')
        factor: 异常值因子
        
    Returns:
        裁剪后的数据
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        return data.clip(lower=lower, upper=upper)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        lower = mean - factor * std
        upper = mean + factor * std
        return data.clip(lower=lower, upper=upper)
    
    else:
        raise ValueError(f"不支持的异常值检测方法: {method}")


def get_feature_names_from_pipeline(pipeline, feature_names: List[str]) -> List[str]:
    """
    从sklearn pipeline获取特征名称
    
    Args:
        pipeline: sklearn pipeline对象
        feature_names: 原始特征名称列表
        
    Returns:
        转换后的特征名称列表
    """
    try:
        # 尝试获取特征名称
        if hasattr(pipeline, 'get_feature_names_out'):
            return list(pipeline.get_feature_names_out(feature_names))
        elif hasattr(pipeline, 'get_feature_names'):
            return list(pipeline.get_feature_names(feature_names))
        else:
            return feature_names
    except:
        return feature_names


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None,
                      check_missing: bool = True, check_duplicates: bool = True) -> Dict[str, Any]:
    """
    验证DataFrame
    
    Args:
        df: 输入DataFrame
        required_columns: 必需的列名列表
        check_missing: 是否检查缺失值
        check_duplicates: 是否检查重复行
        
    Returns:
        验证结果字典
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    # 检查必需列
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"缺少必需列: {missing_cols}")
    
    # 检查缺失值
    if check_missing:
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        if len(missing_cols) > 0:
            results['warnings'].append(f"存在缺失值的列: {missing_cols.to_dict()}")
            results['info']['missing_values'] = missing_cols.to_dict()
    
    # 检查重复行
    if check_duplicates:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results['warnings'].append(f"存在 {duplicates} 行重复数据")
            results['info']['duplicates'] = duplicates
    
    # 基本信息
    results['info']['shape'] = df.shape
    results['info']['dtypes'] = df.dtypes.value_counts().to_dict()
    results['info']['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    return results


def timer(func):
    """
    装饰器：计算函数执行时间
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        print(f"{func.__name__} 执行时间: {execution_time:.2f} 秒")
        return result
    return wrapper
