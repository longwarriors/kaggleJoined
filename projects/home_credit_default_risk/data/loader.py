"""
数据加载模块

负责加载Home Credit Default Risk竞赛的所有数据文件，
支持数据验证、缓存和内存优化。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
import os

try:
    from ..core.base import BaseProcessor
    from ..core.utils import (
        reduce_memory_usage,
        get_memory_usage,
        save_object,
        load_object,
    )
except ImportError:
    # 处理直接运行时的导入问题
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from core.base import BaseProcessor
    from core.utils import (
        reduce_memory_usage,
        get_memory_usage,
        save_object,
        load_object,
    )


class DataLoader(BaseProcessor):
    """
    数据加载器

    负责加载和管理Home Credit Default Risk竞赛的所有数据文件
    """

    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化数据加载器

        Args:
            config: 数据配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.data = {}
        self.data_info = {}

    def fit(self, data: pd.DataFrame = None, **kwargs) -> "DataLoader":
        """
        拟合数据加载器（主要用于设置配置）

        Args:
            data: 不使用，保持接口一致性
            **kwargs: 其他参数

        Returns:
            self
        """
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        加载所有数据

        Args:
            data: 不使用，保持接口一致性
            **kwargs: 其他参数

        Returns:
            包含所有数据的字典
        """
        return self.load_all_data()

    def load_all_data(self, use_cache: bool = None) -> Dict[str, pd.DataFrame]:
        """
        加载所有数据集

        Args:
            use_cache: 是否使用缓存，默认根据配置决定

        Returns:
            包含所有数据的字典
        """
        self._log_info("开始加载所有数据集")

        # 获取配置
        data_paths = self.config.get("data_paths", {})
        caching_config = self.config.get("caching", {})
        memory_config = self.config.get("memory_optimization", {})

        if use_cache is None:
            use_cache = caching_config.get("enable", False)

        # 检查缓存
        if use_cache:
            cached_data = self._load_from_cache()
            if cached_data:
                self._log_info("从缓存加载数据成功")
                self.data = cached_data
                return self.data

        # 逐个加载数据文件
        for dataset_name, file_path in data_paths.items():
            try:
                self._log_info(f"加载数据集: {dataset_name}")

                # 加载数据
                df = self._load_single_file(file_path)

                # 内存优化
                if memory_config.get("reduce_dtypes", True):
                    df = reduce_memory_usage(df, verbose=False)

                self.data[dataset_name] = df

                # 记录数据信息
                self.data_info[dataset_name] = {
                    "shape": df.shape,
                    "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
                    "dtypes": df.dtypes.value_counts().to_dict(),
                    "missing_values": df.isnull().sum().sum(),
                    "file_path": file_path,
                }

                self._log_info(
                    f"成功加载 {dataset_name}, 形状: {df.shape}, "
                    f"内存: {self.data_info[dataset_name]['memory_mb']:.2f} MB"
                )

            except Exception as e:
                self._log_error(f"加载 {dataset_name} 失败: {str(e)}")
                raise

        # 保存到缓存
        if use_cache:
            self._save_to_cache()

        # 内存清理
        if memory_config.get("cleanup_after_processing", True):
            import gc

            gc.collect()

        # 记录总体信息
        total_memory = sum(info["memory_mb"] for info in self.data_info.values())
        self._log_info(f"所有数据加载完成，总内存使用: {total_memory:.2f} MB")

        if memory_config.get("monitor_memory", True):
            memory_usage = get_memory_usage()
            self._log_info(
                f"系统内存使用: {memory_usage['rss_mb']:.2f} MB "
                f"({memory_usage['percent']:.1f}%)"
            )

        return self.data

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            加载的DataFrame
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 根据文件扩展名选择加载方法
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix.lower() in [".pkl", ".pickle"]:
            return pd.read_pickle(file_path)
        else:
            # 默认尝试CSV
            return pd.read_csv(file_path)

    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取训练集和测试集

        Returns:
            训练集和测试集的元组
        """
        if "application_train" not in self.data or "application_test" not in self.data:
            raise ValueError("请先调用load_all_data()方法加载数据")

        return self.data["application_train"], self.data["application_test"]

    def get_auxiliary_data(self) -> Dict[str, pd.DataFrame]:
        """
        获取辅助数据表

        Returns:
            辅助数据表字典
        """
        auxiliary_tables = [
            "bureau",
            "bureau_balance",
            "previous_application",
            "pos_cash_balance",
            "credit_card_balance",
            "installments_payments",
        ]

        return {name: self.data[name] for name in auxiliary_tables if name in self.data}

    def get_data_info(self) -> Dict[str, Dict]:
        """
        获取数据信息

        Returns:
            数据信息字典
        """
        return self.data_info.copy()

    def sample_data(
        self, sample_config: Optional[Dict] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        对数据进行采样（用于开发和测试）

        Args:
            sample_config: 采样配置

        Returns:
            采样后的数据字典
        """
        if sample_config is None:
            sample_config = self.config.get("sampling", {})

        if not sample_config.get("enable_sampling", False):
            return self.data

        self._log_info("开始数据采样")

        sampled_data = {}
        method = sample_config.get("method", "random")
        sample_size = sample_config.get("sample_size", 0.1)
        random_state = sample_config.get("random_state", 42)

        for name, df in self.data.items():
            if name == "application_train" and method == "stratified":
                # 分层采样
                stratify_col = sample_config.get("stratify_column", "TARGET")
                if stratify_col in df.columns:
                    sampled_df = (
                        df.groupby(stratify_col, group_keys=False)
                        .apply(
                            lambda x: x.sample(
                                n=min(int(len(x) * sample_size), len(x)),
                                random_state=random_state,
                            )
                        )
                        .reset_index(drop=True)
                    )
                else:
                    sampled_df = df.sample(frac=sample_size, random_state=random_state)
            else:
                # 随机采样
                if sample_size < 1:
                    sampled_df = df.sample(frac=sample_size, random_state=random_state)
                else:
                    sampled_df = df.sample(
                        n=min(int(sample_size), len(df)), random_state=random_state
                    )

            sampled_data[name] = sampled_df
            self._log_info(f"{name} 采样: {df.shape} -> {sampled_df.shape}")

        return sampled_data

    def _load_from_cache(self) -> Optional[Dict[str, pd.DataFrame]]:
        """从缓存加载数据"""
        try:
            cache_config = self.config.get("caching", {})
            cache_dir = Path(cache_config.get("cache_dir", "./cache/data"))
            cache_file = cache_dir / "all_data.pkl"

            if cache_file.exists():
                # 检查缓存是否过期
                import time

                file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                expiry_hours = cache_config.get("expiry_hours", 24)

                if file_age_hours < expiry_hours:
                    return load_object(cache_file, method="pickle")
                else:
                    self._log_info("缓存已过期，将重新加载数据")

        except Exception as e:
            self._log_warning(f"从缓存加载数据失败: {e}")

        return None

    def _save_to_cache(self):
        """保存数据到缓存"""
        try:
            cache_config = self.config.get("caching", {})
            cache_dir = Path(cache_config.get("cache_dir", "./cache/data"))
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_file = cache_dir / "all_data.pkl"
            save_object(self.data, cache_file, method="pickle")

            self._log_info(f"数据已保存到缓存: {cache_file}")

        except Exception as e:
            self._log_warning(f"保存数据到缓存失败: {e}")

    def validate_data_paths(self) -> Dict[str, bool]:
        """
        验证数据文件路径

        Returns:
            验证结果字典
        """
        data_paths = self.config.get("data_paths", {})
        results = {}

        for name, path in data_paths.items():
            file_path = Path(path)
            results[name] = file_path.exists()

            if not results[name]:
                self._log_warning(f"数据文件不存在: {name} -> {path}")

        return results
