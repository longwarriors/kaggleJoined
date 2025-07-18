"""
主流水线模块

整合所有组件，提供完整的端到端机器学习流水线。

作者：Augment Agent
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from ..core.base import BasePipeline
from ..core.config import ConfigManager
from ..core.logger import LoggerManager
from ..core.utils import get_timestamp, save_json

from ..data.loader import DataLoader
from ..data.eda import EDAAnalyzer
from ..data.cleaner import DataCleaner
from ..data.validator import DataValidator

from ..features.builder import FeatureBuilder
from ..features.selector import FeatureSelector

from ..models.baseline import BaselineModel
from ..models.trainers import ModelTrainer
from ..models.ensemble import EnsembleModel
from ..models.evaluator import ModelEvaluator
from ..models.optimizer import HyperparameterOptimizer
from ..models.serializer import ModelSerializer


class HomeCreditPipeline(BasePipeline):
    """
    Home Credit Default Risk 主流水线

    整合数据处理、特征工程、模型训练、集成建模的完整流水线
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化主流水线

        Args:
            config_dir: 配置文件目录路径
        """
        # 初始化配置和日志
        self.config_manager = ConfigManager()
        self.logger_manager = LoggerManager()

        # 加载配置
        if config_dir:
            self.configs = self.config_manager.load_all_configs(config_dir)
        else:
            self.configs = self.config_manager.load_all_configs()

        # 设置日志
        self.logger = self.logger_manager.setup_pipeline_logger("home_credit_pipeline")

        super().__init__(self.configs, self.logger)

        # 初始化组件
        self.data_loader = None
        self.data_validator = None
        self.eda_analyzer = None
        self.data_cleaner = None
        self.feature_builder = None
        self.feature_selector = None
        self.baseline_model = None
        self.model_trainer = None
        self.ensemble_model = None
        self.evaluator = None
        self.optimizer = None
        self.model_serializer = None

        # 存储中间结果
        self.raw_data = {}
        self.cleaned_data = {}
        self.features_df = None
        self.selected_features_df = None
        self.target = None

        # 存储最终结果
        self.pipeline_results = {}

    def run(self, **kwargs) -> Dict:
        """
        运行完整流水线

        Args:
            **kwargs: 其他参数

        Returns:
            流水线运行结果
        """
        self._log_info("=" * 60)
        self._log_info("Home Credit Default Risk Pipeline 开始运行")
        self._log_info("=" * 60)

        try:
            # 1. 数据加载和验证
            self._log_info("阶段 1: 数据加载和验证")
            self._load_and_validate_data()

            # 2. 探索性数据分析
            self._log_info("阶段 2: 探索性数据分析")
            eda_results = self._run_eda()

            # 3. 数据清洗
            self._log_info("阶段 3: 数据清洗")
            self._clean_data()

            # 4. 特征工程
            self._log_info("阶段 4: 特征工程")
            self._build_features()

            # 5. 特征选择
            self._log_info("阶段 5: 特征选择")
            self._select_features()

            # 6. 基线模型
            self._log_info("阶段 6: 基线模型建立")
            baseline_results = self._train_baseline_models()

            # 7. 高级模型训练
            self._log_info("阶段 7: 高级模型训练")
            model_results = self._train_advanced_models()

            # 8. 超参数优化（可选）
            if self.config.get("hyperparameter_tuning", {}).get("enable", False):
                self._log_info("阶段 8: 超参数优化")
                optimization_results = self._optimize_hyperparameters()
            else:
                optimization_results = {}

            # 9. 集成建模
            self._log_info("阶段 9: 集成建模")
            ensemble_results = self._train_ensemble_models()

            # 10. 最终评估和报告
            self._log_info("阶段 10: 最终评估")
            final_results = self._final_evaluation()

            # 整合所有结果
            self.pipeline_results = {
                "eda_results": eda_results,
                "baseline_results": baseline_results,
                "model_results": model_results,
                "optimization_results": optimization_results,
                "ensemble_results": ensemble_results,
                "final_results": final_results,
                "pipeline_metadata": {
                    "timestamp": get_timestamp(),
                    "data_shape": (
                        self.features_df.shape
                        if self.features_df is not None
                        else (0, 0)
                    ),
                    "selected_features_count": (
                        self.selected_features_df.shape[1]
                        if self.selected_features_df is not None
                        else 0
                    ),
                },
            }

            # 保存结果
            self._save_pipeline_results()

            self._log_info("=" * 60)
            self._log_info("Pipeline 运行完成")
            self._log_info("=" * 60)

            return self.pipeline_results

        except Exception as e:
            self._log_error(f"Pipeline 运行失败: {e}")
            raise

    def _load_and_validate_data(self):
        """加载和验证数据"""
        # 数据加载
        data_config = self.config_manager.get_data_config()
        self.data_loader = DataLoader(data_config, self.logger)

        # 检查数据文件是否存在
        path_validation = self.data_loader.validate_data_paths()
        missing_files = [name for name, exists in path_validation.items() if not exists]

        if missing_files:
            self._log_warning(f"缺少数据文件: {missing_files}")
            # 这里可以选择使用模拟数据或者抛出异常
            raise FileNotFoundError(f"缺少必要的数据文件: {missing_files}")

        # 加载数据
        self.raw_data = self.data_loader.load_all_data()

        # 数据验证
        self.data_validator = DataValidator(data_config, self.logger)
        validation_results = self.data_validator.fit_transform(self.raw_data)

        # 检查验证结果
        summary = self.data_validator.get_validation_summary()
        if not summary.get("overall_passed", False):
            self._log_warning("数据验证未完全通过，请检查数据质量")

    def _run_eda(self) -> Dict:
        """运行探索性数据分析"""
        data_config = self.config_manager.get_data_config()
        self.eda_analyzer = EDAAnalyzer(data_config, self.logger)

        eda_results = self.eda_analyzer.fit_transform(self.raw_data)

        # 生成EDA报告
        self.eda_analyzer.generate_eda_report()

        return eda_results

    def _clean_data(self):
        """清洗数据"""
        data_config = self.config_manager.get_data_config()
        self.data_cleaner = DataCleaner(data_config, self.logger)

        self.cleaned_data = self.data_cleaner.fit_transform(self.raw_data)

        # 记录清洗统计
        cleaning_stats = self.data_cleaner.get_cleaning_summary()
        self._log_info(
            f"数据清洗完成，处理了 {cleaning_stats.get('total_datasets', 0)} 个数据集"
        )

    def _build_features(self):
        """构建特征"""
        feature_config = self.config_manager.get_feature_config()
        self.feature_builder = FeatureBuilder(feature_config, self.logger)

        # 获取目标变量
        if (
            "application_train" in self.cleaned_data
            and "TARGET" in self.cleaned_data["application_train"].columns
        ):
            self.target = self.cleaned_data["application_train"]["TARGET"]

        # 构建特征
        self.features_df = self.feature_builder.fit_transform(
            self.cleaned_data, target=self.target
        )

        # 记录特征统计
        feature_stats = self.feature_builder.get_feature_stats()
        self._log_info(
            f"特征工程完成，生成了 {feature_stats.get('final_features', 0)} 个特征"
        )

    def _select_features(self):
        """选择特征"""
        feature_config = self.config_manager.get_feature_config()
        self.feature_selector = FeatureSelector(feature_config, self.logger)

        # 特征选择
        if self.target is not None:
            # 对齐数据
            common_idx = self.features_df.index.intersection(self.target.index)
            features_aligned = self.features_df.loc[common_idx]
            target_aligned = self.target.loc[common_idx]

            self.selected_features_df = self.feature_selector.fit_transform(
                features_aligned, target_aligned
            )
        else:
            self.selected_features_df = self.feature_selector.fit_transform(
                self.features_df
            )

        # 记录选择统计
        selection_stats = self.feature_selector.get_selection_stats()
        self._log_info(
            f"特征选择完成，选择了 {selection_stats.get('selected_features', 0)} 个特征"
        )

    def _train_baseline_models(self) -> Dict:
        """训练基线模型"""
        model_config = self.config_manager.get_model_config()
        self.baseline_model = BaselineModel(model_config, self.logger)

        if self.target is not None:
            # 对齐数据
            common_idx = self.selected_features_df.index.intersection(self.target.index)
            X_train = self.selected_features_df.loc[common_idx]
            y_train = self.target.loc[common_idx]

            # 训练基线模型
            baseline_results = self.baseline_model.train(X_train, y_train)
            baseline_score = self.baseline_model.establish_baseline()

            self._log_info(f"基线模型训练完成，基线分数: {baseline_score:.4f}")
            return baseline_results
        else:
            self._log_warning("无目标变量，跳过基线模型训练")
            return {}

    def _train_advanced_models(self) -> Dict:
        """训练高级模型"""
        model_config = self.config_manager.get_model_config()
        self.model_trainer = ModelTrainer(model_config, self.logger)

        if self.target is not None:
            # 对齐数据
            common_idx = self.selected_features_df.index.intersection(self.target.index)
            X_train = self.selected_features_df.loc[common_idx]
            y_train = self.target.loc[common_idx]

            # 训练高级模型
            model_results = self.model_trainer.train(X_train, y_train)

            best_model_name, _, best_score = self.model_trainer.get_best_model()
            self._log_info(
                f"高级模型训练完成，最佳模型: {best_model_name} (AUC: {best_score:.4f})"
            )

            # 保存训练好的模型
            self._save_trained_models(X_train, model_results)

            return model_results
        else:
            self._log_warning("无目标变量，跳过高级模型训练")
            return {}

    def _optimize_hyperparameters(self) -> Dict:
        """优化超参数"""
        model_config = self.config_manager.get_model_config()
        self.optimizer = HyperparameterOptimizer(model_config, self.logger)

        if self.target is not None:
            # 对齐数据
            common_idx = self.selected_features_df.index.intersection(self.target.index)
            X_train = self.selected_features_df.loc[common_idx]
            y_train = self.target.loc[common_idx]

            # 超参数优化
            optimization_results = self.optimizer.fit_transform((X_train, y_train))

            # 更新模型配置
            updated_config = self.optimizer.update_model_config(model_config)
            self.config_manager.update_config("model_config", updated_config)

            return optimization_results
        else:
            self._log_warning("无目标变量，跳过超参数优化")
            return {}

    def _train_ensemble_models(self) -> Dict:
        """训练集成模型"""
        model_config = self.config_manager.get_model_config()
        self.ensemble_model = EnsembleModel(model_config, self.logger)

        if self.target is not None and self.model_trainer is not None:
            # 对齐数据
            common_idx = self.selected_features_df.index.intersection(self.target.index)
            X_train = self.selected_features_df.loc[common_idx]
            y_train = self.target.loc[common_idx]

            # 训练集成模型
            ensemble_results = self.ensemble_model.train(
                X_train, y_train, trained_models=self.model_trainer.trained_models
            )

            best_ensemble_name, _, best_score = self.ensemble_model.get_best_ensemble()
            self._log_info(
                f"集成模型训练完成，最佳集成: {best_ensemble_name} (AUC: {best_score:.4f})"
            )

            # 保存集成模型
            self._save_ensemble_models(X_train, ensemble_results)

            return ensemble_results
        else:
            self._log_warning("无目标变量或基础模型，跳过集成建模")
            return {}

    def _final_evaluation(self) -> Dict:
        """最终评估"""
        model_config = self.config_manager.get_model_config()
        self.evaluator = ModelEvaluator(model_config, self.logger)

        final_results = {}

        if self.target is not None:
            # 对齐数据
            common_idx = self.selected_features_df.index.intersection(self.target.index)
            X_eval = self.selected_features_df.loc[common_idx]
            y_eval = self.target.loc[common_idx]

            # 评估最佳模型
            if self.ensemble_model and self.ensemble_model.is_trained:
                # 使用最佳集成模型
                y_pred = self.ensemble_model.predict(X_eval)
                y_pred_proba = self.ensemble_model.predict_proba(X_eval)

                eval_results = self.evaluator.evaluate(y_eval, y_pred, y_pred_proba)
                final_results["best_model_type"] = "ensemble"
                final_results["best_model_name"] = (
                    self.ensemble_model.best_ensemble_name
                )
                final_results["evaluation"] = eval_results

            elif self.model_trainer and self.model_trainer.is_trained:
                # 使用最佳高级模型
                y_pred = self.model_trainer.predict(X_eval)
                y_pred_proba = self.model_trainer.predict_proba(X_eval)

                eval_results = self.evaluator.evaluate(y_eval, y_pred, y_pred_proba)
                final_results["best_model_type"] = "advanced"
                final_results["best_model_name"] = self.model_trainer.best_model_name
                final_results["evaluation"] = eval_results

            elif self.baseline_model and self.baseline_model.is_trained:
                # 使用基线模型
                y_pred = self.baseline_model.predict(X_eval)
                y_pred_proba = self.baseline_model.predict_proba(X_eval)

                eval_results = self.evaluator.evaluate(y_eval, y_pred, y_pred_proba)
                final_results["best_model_type"] = "baseline"
                final_results["best_model_name"] = (
                    self.baseline_model.best_baseline_model
                )
                final_results["evaluation"] = eval_results

            # 与基线比较
            if self.baseline_model and self.baseline_model.is_trained:
                baseline_score = self.baseline_model.best_baseline_score
                current_score = eval_results.get("roc_auc", 0)

                comparison = self.baseline_model.compare_with_baseline(
                    current_score, final_results.get("best_model_name", "Unknown")
                )
                final_results["baseline_comparison"] = comparison

        return final_results

    def _save_pipeline_results(self):
        """保存流水线结果"""
        output_dir = Path("outputs/pipeline_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = get_timestamp()
        results_file = output_dir / f"pipeline_results_{timestamp}.json"

        save_json(self.pipeline_results, results_file)
        self._log_info(f"流水线结果已保存到: {results_file}")

    def generate_submission(self, test_data_path: Optional[str] = None) -> str:
        """
        生成Kaggle提交文件

        Args:
            test_data_path: 测试数据路径

        Returns:
            提交文件路径
        """
        if not self.is_trained():
            raise ValueError("请先运行完整流水线")

        # 加载测试数据（如果提供路径）
        if test_data_path:
            test_df = pd.read_csv(test_data_path)
        elif "application_test" in self.cleaned_data:
            test_df = self.cleaned_data["application_test"]
        else:
            raise ValueError("未找到测试数据")

        # 特征工程
        test_features = self.feature_builder.transform({"application_test": test_df})
        test_selected = self.feature_selector.transform(test_features)

        # 预测
        if self.ensemble_model and self.ensemble_model.is_trained:
            predictions = self.ensemble_model.predict_proba(test_selected)
        elif self.model_trainer and self.model_trainer.is_trained:
            predictions = self.model_trainer.predict_proba(test_selected)
        else:
            predictions = self.baseline_model.predict_proba(test_selected)

        # 创建提交文件
        submission = pd.DataFrame(
            {"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": predictions}
        )

        # 保存提交文件
        output_dir = Path("outputs/submissions")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = get_timestamp()
        submission_file = output_dir / f"submission_{timestamp}.csv"
        submission.to_csv(submission_file, index=False)

        self._log_info(f"提交文件已生成: {submission_file}")
        return str(submission_file)

    def is_trained(self) -> bool:
        """检查是否已训练"""
        return (
            (self.baseline_model and self.baseline_model.is_trained)
            or (self.model_trainer and self.model_trainer.is_trained)
            or (self.ensemble_model and self.ensemble_model.is_trained)
        )

    def _save_trained_models(self, X_train: pd.DataFrame, model_results: Dict):
        """保存训练好的模型"""
        try:
            # 初始化模型序列化器
            if self.model_serializer is None:
                self.model_serializer = ModelSerializer(self.configs, self.logger)

            # 获取特征名称
            feature_names = X_train.columns.tolist()

            # 保存每个训练好的模型
            for model_name, model_data in self.model_trainer.trained_models.items():
                model = model_data["model"]
                cv_scores = model_data.get("cv_scores", [])

                # 准备元数据
                metadata = {
                    "cv_scores": cv_scores,
                    "cv_mean": np.mean(cv_scores) if cv_scores else 0.0,
                    "cv_std": np.std(cv_scores) if cv_scores else 0.0,
                    "training_samples": len(X_train),
                    "feature_count": len(feature_names),
                    "model_params": getattr(model, "get_params", lambda: {})(),
                    "performance": {"cv_auc": np.mean(cv_scores) if cv_scores else 0.0},
                }

                # 确定模型类型
                model_type = self._get_model_type(model_name)

                # 保存模型
                model_path = self.model_serializer.save_model(
                    model=model,
                    model_name=model_name,
                    model_type=model_type,
                    metadata=metadata,
                    feature_names=feature_names,
                )

                self._log_info(f"模型已保存: {model_name} -> {model_path}")

            # 保存特征工程器
            self._save_feature_builder()

        except Exception as e:
            self._log_error(f"保存模型失败: {e}")

    def _save_ensemble_models(self, X_train: pd.DataFrame, ensemble_results: Dict):
        """保存集成模型"""
        try:
            # 初始化模型序列化器
            if self.model_serializer is None:
                self.model_serializer = ModelSerializer(self.configs, self.logger)

            # 获取特征名称
            feature_names = X_train.columns.tolist()

            # 保存每个集成模型
            for (
                ensemble_name,
                ensemble_data,
            ) in self.ensemble_model.trained_ensembles.items():
                ensemble = ensemble_data["ensemble"]
                cv_scores = ensemble_data.get("cv_scores", [])

                # 准备元数据
                metadata = {
                    "cv_scores": cv_scores,
                    "cv_mean": np.mean(cv_scores) if cv_scores else 0.0,
                    "cv_std": np.std(cv_scores) if cv_scores else 0.0,
                    "training_samples": len(X_train),
                    "feature_count": len(feature_names),
                    "ensemble_type": ensemble_name,
                    "base_models": (
                        list(self.model_trainer.trained_models.keys())
                        if self.model_trainer
                        else []
                    ),
                    "performance": {"cv_auc": np.mean(cv_scores) if cv_scores else 0.0},
                }

                # 保存集成模型
                model_path = self.model_serializer.save_model(
                    model=ensemble,
                    model_name=f"ensemble_{ensemble_name}",
                    model_type="ensemble",
                    metadata=metadata,
                    feature_names=feature_names,
                )

                self._log_info(f"集成模型已保存: {ensemble_name} -> {model_path}")

        except Exception as e:
            self._log_error(f"保存集成模型失败: {e}")

    def _save_feature_builder(self):
        """保存特征工程器"""
        try:
            import pickle

            # 确定保存路径
            output_dir = self.configs.get("output", {}).get("base_dir", "./outputs")
            feature_builder_path = os.path.join(output_dir, "feature_builder.pkl")

            # 创建目录
            os.makedirs(output_dir, exist_ok=True)

            # 保存特征工程器
            with open(feature_builder_path, "wb") as f:
                pickle.dump(self.feature_builder, f)

            self._log_info(f"特征工程器已保存: {feature_builder_path}")

        except Exception as e:
            self._log_error(f"保存特征工程器失败: {e}")

    def _get_model_type(self, model_name: str) -> str:
        """根据模型名称确定模型类型"""
        model_name_lower = model_name.lower()

        if "lightgbm" in model_name_lower or "lgb" in model_name_lower:
            return "lightgbm"
        elif "xgboost" in model_name_lower or "xgb" in model_name_lower:
            return "xgboost"
        elif "catboost" in model_name_lower or "cat" in model_name_lower:
            return "catboost"
        elif "random_forest" in model_name_lower or "rf" in model_name_lower:
            return "sklearn"
        elif "logistic" in model_name_lower:
            return "sklearn"
        else:
            return "unknown"

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """获取流水线摘要"""
        summary = {
            "pipeline_status": "completed" if self.pipeline_results else "not_run",
            "data_loaded": bool(self.raw_data),
            "features_built": self.features_df is not None,
            "features_selected": self.selected_features_df is not None,
            "baseline_trained": self.baseline_model and self.baseline_model.is_trained,
            "advanced_models_trained": self.model_trainer
            and self.model_trainer.is_trained,
            "ensemble_trained": self.ensemble_model and self.ensemble_model.is_trained,
        }

        if self.pipeline_results:
            final_results = self.pipeline_results.get("final_results", {})
            summary.update(
                {
                    "best_model_type": final_results.get("best_model_type"),
                    "best_model_name": final_results.get("best_model_name"),
                    "final_auc": final_results.get("evaluation", {}).get("roc_auc", 0),
                }
            )

        return summary
