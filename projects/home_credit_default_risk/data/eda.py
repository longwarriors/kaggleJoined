"""
探索性数据分析模块

提供全面的EDA功能，包括数据概览、分布分析、相关性分析、
缺失值分析等，帮助理解数据特征和发现潜在问题。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    from ..core.base import BaseProcessor
    from ..core.utils import get_timestamp, create_directory
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from core.base import BaseProcessor
    from core.utils import get_timestamp, create_directory


class EDAAnalyzer(BaseProcessor):
    """
    探索性数据分析器

    提供全面的EDA分析功能
    """

    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化EDA分析器

        Args:
            config: 配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.analysis_results = {}
        self.plots_dir = Path("outputs/eda_plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # 设置绘图样式
        plt.style.use("default")
        sns.set_palette("husl")

    def fit(self, data: Dict[str, pd.DataFrame], **kwargs) -> "EDAAnalyzer":
        """
        拟合EDA分析器

        Args:
            data: 数据字典
            **kwargs: 其他参数

        Returns:
            self
        """
        self.data = data
        self.is_fitted = True
        return self

    def transform(
        self, data: Dict[str, pd.DataFrame] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        执行EDA分析

        Args:
            data: 数据字典，如果不提供则使用fit时的数据
            **kwargs: 其他参数

        Returns:
            EDA分析结果字典
        """
        if data is not None:
            self.data = data

        return self.analyze_all()

    def analyze_all(
        self, save_plots: bool = True, save_report: bool = True
    ) -> Dict[str, Any]:
        """
        执行完整的EDA分析

        Args:
            save_plots: 是否保存图表
            save_report: 是否保存报告

        Returns:
            完整的EDA分析结果
        """
        self._log_info("开始执行完整EDA分析")

        results = {}

        # 1. 数据概览
        self._log_info("执行数据概览分析")
        results["overview"] = self.analyze_data_overview()

        # 2. 目标变量分析（如果存在）
        if (
            "application_train" in self.data
            and "TARGET" in self.data["application_train"].columns
        ):
            self._log_info("执行目标变量分析")
            results["target_analysis"] = self.analyze_target_variable()

        # 3. 数值特征分析
        self._log_info("执行数值特征分析")
        results["numerical_analysis"] = self.analyze_numerical_features()

        # 4. 类别特征分析
        self._log_info("执行类别特征分析")
        results["categorical_analysis"] = self.analyze_categorical_features()

        # 5. 缺失值分析
        self._log_info("执行缺失值分析")
        results["missing_analysis"] = self.analyze_missing_values()

        # 6. 相关性分析
        self._log_info("执行相关性分析")
        results["correlation_analysis"] = self.analyze_correlations()

        # 7. 异常值分析
        self._log_info("执行异常值分析")
        results["outlier_analysis"] = self.analyze_outliers()

        # 8. 数据质量分析
        self._log_info("执行数据质量分析")
        results["quality_analysis"] = self.analyze_data_quality()

        self.analysis_results = results

        # 保存报告
        if save_report:
            self.generate_eda_report()

        self._log_info("EDA分析完成")
        return results

    def analyze_data_overview(self) -> Dict[str, Any]:
        """
        分析数据概览

        Returns:
            数据概览分析结果
        """
        overview = {}

        for name, df in self.data.items():
            info = {
                "shape": df.shape,
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "dtypes_count": df.dtypes.value_counts().to_dict(),
                "missing_values_count": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
                "duplicate_rows": df.duplicated().sum(),
                "unique_values_per_column": df.nunique().to_dict(),
            }

            # 数值列和类别列统计
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            info["numerical_columns_count"] = len(numerical_cols)
            info["categorical_columns_count"] = len(categorical_cols)
            info["numerical_columns"] = numerical_cols[:10]  # 只显示前10个
            info["categorical_columns"] = categorical_cols[:10]

            overview[name] = info

        return overview

    def analyze_target_variable(self) -> Dict[str, Any]:
        """
        分析目标变量

        Returns:
            目标变量分析结果
        """
        if "application_train" not in self.data:
            return {}

        df = self.data["application_train"]
        target_col = "TARGET"

        if target_col not in df.columns:
            return {}

        target_analysis = {}

        # 基本统计
        target_analysis["value_counts"] = df[target_col].value_counts().to_dict()
        target_analysis["percentage"] = (
            df[target_col].value_counts(normalize=True) * 100
        ).to_dict()
        target_analysis["missing_count"] = df[target_col].isnull().sum()

        # 类别不平衡分析
        class_counts = df[target_col].value_counts()
        if len(class_counts) == 2:
            imbalance_ratio = class_counts.min() / class_counts.max()
            target_analysis["imbalance_ratio"] = imbalance_ratio
            target_analysis["is_imbalanced"] = imbalance_ratio < 0.1

        return target_analysis

    def analyze_numerical_features(self) -> Dict[str, Any]:
        """
        分析数值特征

        Returns:
            数值特征分析结果
        """
        numerical_analysis = {}

        for name, df in self.data.items():
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if not numerical_cols:
                continue

            # 基本统计信息
            desc_stats = df[numerical_cols].describe()

            # 偏度和峰度
            skewness = df[numerical_cols].skew()
            kurtosis = df[numerical_cols].kurtosis()

            # 零值和负值统计
            zero_counts = (df[numerical_cols] == 0).sum()
            negative_counts = (df[numerical_cols] < 0).sum()

            numerical_analysis[name] = {
                "descriptive_stats": desc_stats.to_dict(),
                "skewness": skewness.to_dict(),
                "kurtosis": kurtosis.to_dict(),
                "zero_counts": zero_counts.to_dict(),
                "negative_counts": negative_counts.to_dict(),
                "highly_skewed_features": skewness[abs(skewness) > 2].index.tolist(),
            }

        return numerical_analysis

    def analyze_categorical_features(self) -> Dict[str, Any]:
        """
        分析类别特征

        Returns:
            类别特征分析结果
        """
        categorical_analysis = {}

        for name, df in self.data.items():
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            if not categorical_cols:
                continue

            cat_info = {}

            for col in categorical_cols:
                unique_count = df[col].nunique()
                value_counts = df[col].value_counts()

                cat_info[col] = {
                    "unique_count": unique_count,
                    "most_frequent": (
                        value_counts.index[0] if len(value_counts) > 0 else None
                    ),
                    "most_frequent_count": (
                        value_counts.iloc[0] if len(value_counts) > 0 else 0
                    ),
                    "cardinality_level": self._get_cardinality_level(
                        unique_count, len(df)
                    ),
                    "top_5_values": value_counts.head().to_dict(),
                }

            categorical_analysis[name] = cat_info

        return categorical_analysis

    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        分析缺失值

        Returns:
            缺失值分析结果
        """
        missing_analysis = {}

        for name, df in self.data.items():
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100

            # 只保留有缺失值的列
            missing_cols = missing_counts[missing_counts > 0]

            if len(missing_cols) == 0:
                missing_analysis[name] = {"has_missing": False}
                continue

            # 缺失值模式分析
            missing_patterns = df[missing_cols.index].isnull().value_counts()

            missing_analysis[name] = {
                "has_missing": True,
                "missing_counts": missing_cols.to_dict(),
                "missing_percentages": missing_percentages[
                    missing_cols.index
                ].to_dict(),
                "columns_with_high_missing": missing_percentages[
                    missing_percentages > 50
                ].index.tolist(),
                "total_missing_values": missing_counts.sum(),
                "missing_patterns_count": len(missing_patterns),
            }

        return missing_analysis

    def analyze_correlations(self) -> Dict[str, Any]:
        """
        分析特征相关性

        Returns:
            相关性分析结果
        """
        correlation_analysis = {}

        for name, df in self.data.items():
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numerical_cols) < 2:
                continue

            # 计算相关性矩阵
            corr_matrix = df[numerical_cols].corr()

            # 找出高相关性特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:  # 高相关性阈值
                        high_corr_pairs.append(
                            {
                                "feature1": corr_matrix.columns[i],
                                "feature2": corr_matrix.columns[j],
                                "correlation": corr_value,
                            }
                        )

            correlation_analysis[name] = {
                "correlation_matrix_shape": corr_matrix.shape,
                "high_correlation_pairs": high_corr_pairs,
                "max_correlation": corr_matrix.abs().max().max(),
                "mean_correlation": corr_matrix.abs().mean().mean(),
            }

            # 如果是训练数据且有目标变量，分析与目标变量的相关性
            if name == "application_train" and "TARGET" in df.columns:
                target_corr = (
                    df[numerical_cols]
                    .corrwith(df["TARGET"])
                    .abs()
                    .sort_values(ascending=False)
                )
                correlation_analysis[name]["target_correlations"] = target_corr.head(
                    10
                ).to_dict()

        return correlation_analysis

    def analyze_outliers(self) -> Dict[str, Any]:
        """
        分析异常值

        Returns:
            异常值分析结果
        """
        outlier_analysis = {}

        for name, df in self.data.items():
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if not numerical_cols:
                continue

            outlier_info = {}

            for col in numerical_cols:
                # IQR方法检测异常值
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

                outlier_info[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": (len(outliers) / len(df)) * 100,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "has_outliers": len(outliers) > 0,
                }

            outlier_analysis[name] = outlier_info

        return outlier_analysis

    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        分析数据质量

        Returns:
            数据质量分析结果
        """
        quality_analysis = {}

        for name, df in self.data.items():
            quality_score = 100  # 初始质量分数
            issues = []

            # 检查缺失值
            missing_percentage = (df.isnull().sum().sum() / df.size) * 100
            if missing_percentage > 10:
                quality_score -= 20
                issues.append(f"高缺失值比例: {missing_percentage:.2f}%")

            # 检查重复行
            duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
            if duplicate_percentage > 5:
                quality_score -= 15
                issues.append(f"高重复行比例: {duplicate_percentage:.2f}%")

            # 检查数据类型一致性
            mixed_types = []
            for col in df.columns:
                if df[col].dtype == "object":
                    # 检查是否包含数字字符串
                    try:
                        pd.to_numeric(df[col].dropna().head(100))
                        mixed_types.append(col)
                    except:
                        pass

            if mixed_types:
                quality_score -= 10
                issues.append(f"可能的数据类型不一致: {len(mixed_types)} 列")

            quality_analysis[name] = {
                "quality_score": max(0, quality_score),
                "issues": issues,
                "missing_percentage": missing_percentage,
                "duplicate_percentage": duplicate_percentage,
                "mixed_type_columns": mixed_types,
            }

        return quality_analysis

    def _get_cardinality_level(self, unique_count: int, total_count: int) -> str:
        """
        获取基数水平

        Args:
            unique_count: 唯一值数量
            total_count: 总数量

        Returns:
            基数水平字符串
        """
        ratio = unique_count / total_count

        if ratio > 0.9:
            return "very_high"
        elif ratio > 0.5:
            return "high"
        elif ratio > 0.1:
            return "medium"
        else:
            return "low"

    def generate_eda_report(self, output_path: Optional[str] = None) -> str:
        """
        生成EDA报告

        Args:
            output_path: 输出路径

        Returns:
            报告文件路径
        """
        if output_path is None:
            timestamp = get_timestamp()
            output_path = f"outputs/reports/eda_report_{timestamp}.md"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成Markdown报告
        report_content = self._generate_markdown_report()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        self._log_info(f"EDA报告已保存到: {output_path}")
        return str(output_path)

    def _generate_markdown_report(self) -> str:
        """生成Markdown格式的报告"""
        report = "# Home Credit Default Risk - EDA分析报告\n\n"
        report += f"生成时间: {get_timestamp('%Y-%m-%d %H:%M:%S')}\n\n"

        if "overview" in self.analysis_results:
            report += "## 数据概览\n\n"
            for name, info in self.analysis_results["overview"].items():
                report += f"### {name}\n"
                report += f"- 数据形状: {info['shape']}\n"
                report += f"- 内存使用: {info['memory_usage_mb']:.2f} MB\n"
                report += f"- 缺失值: {info['missing_values_count']} ({info['missing_percentage']:.2f}%)\n"
                report += f"- 重复行: {info['duplicate_rows']}\n"
                report += f"- 数值列: {info['numerical_columns_count']}\n"
                report += f"- 类别列: {info['categorical_columns_count']}\n\n"

        if "target_analysis" in self.analysis_results:
            report += "## 目标变量分析\n\n"
            target_info = self.analysis_results["target_analysis"]
            report += f"- 类别分布: {target_info.get('value_counts', {})}\n"
            report += f"- 百分比: {target_info.get('percentage', {})}\n"
            if "imbalance_ratio" in target_info:
                report += f"- 不平衡比例: {target_info['imbalance_ratio']:.4f}\n"
                report += f"- 是否不平衡: {target_info['is_imbalanced']}\n\n"

        if "missing_analysis" in self.analysis_results:
            report += "## 缺失值分析\n\n"
            for name, info in self.analysis_results["missing_analysis"].items():
                if info.get("has_missing", False):
                    report += f"### {name}\n"
                    report += f"- 总缺失值: {info.get('total_missing_values', 0)}\n"
                    report += f"- 高缺失列: {len(info.get('columns_with_high_missing', []))}\n\n"

        if "quality_analysis" in self.analysis_results:
            report += "## 数据质量分析\n\n"
            for name, info in self.analysis_results["quality_analysis"].items():
                report += f"### {name}\n"
                report += f"- 质量分数: {info['quality_score']}/100\n"
                if info["issues"]:
                    report += f"- 问题: {'; '.join(info['issues'])}\n\n"
                else:
                    report += "- 无明显质量问题\n\n"

        return report

    def plot_target_distribution(self, save_plot: bool = True) -> Optional[str]:
        """
        绘制目标变量分布图

        Args:
            save_plot: 是否保存图表

        Returns:
            图表文件路径（如果保存）
        """
        if (
            "application_train" not in self.data
            or "TARGET" not in self.data["application_train"].columns
        ):
            self._log_warning("无法绘制目标变量分布图：数据或目标列不存在")
            return None

        df = self.data["application_train"]
        target_col = "TARGET"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 计数图
        df[target_col].value_counts().plot(kind="bar", ax=axes[0])
        axes[0].set_title("目标变量分布（计数）")
        axes[0].set_xlabel("目标值")
        axes[0].set_ylabel("计数")

        # 饼图
        df[target_col].value_counts().plot(kind="pie", ax=axes[1], autopct="%1.1f%%")
        axes[1].set_title("目标变量分布（百分比）")
        axes[1].set_ylabel("")

        plt.tight_layout()

        if save_plot:
            plot_path = self.plots_dir / f"target_distribution_{get_timestamp()}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            self._log_info(f"目标变量分布图已保存: {plot_path}")
            return str(plot_path)
        else:
            plt.show()
            return None

    def plot_missing_values_heatmap(
        self, dataset_name: str = "application_train", save_plot: bool = True
    ) -> Optional[str]:
        """
        绘制缺失值热力图

        Args:
            dataset_name: 数据集名称
            save_plot: 是否保存图表

        Returns:
            图表文件路径（如果保存）
        """
        if dataset_name not in self.data:
            self._log_warning(f"数据集 {dataset_name} 不存在")
            return None

        df = self.data[dataset_name]
        missing_data = df.isnull()

        # 只显示有缺失值的列
        missing_cols = missing_data.sum()[missing_data.sum() > 0].index

        if len(missing_cols) == 0:
            self._log_info(f"{dataset_name} 无缺失值")
            return None

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            missing_data[missing_cols].head(1000),
            cbar=True,
            yticklabels=False,
            cmap="viridis",
        )
        plt.title(f"{dataset_name} - 缺失值模式（前1000行）")
        plt.xlabel("特征")
        plt.ylabel("样本")

        if save_plot:
            plot_path = (
                self.plots_dir / f"missing_heatmap_{dataset_name}_{get_timestamp()}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            self._log_info(f"缺失值热力图已保存: {plot_path}")
            return str(plot_path)
        else:
            plt.show()
            return None
