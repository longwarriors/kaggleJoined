"""
DRW 加密市场预测比赛 - 预测与提交
在测试集上生成预测，格式化提交文件
"""

import numpy as np
import pandas as pd
import os
import gc
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, List
import warnings

warnings.filterwarnings("ignore")

# 自定义模块
from data_preprocessing import CryptoDataProcessor
from gradient_boosting_models import GradientBoostingTrainer, EnsembleTrainer
from lstm_model import LSTMTrainer
from model_evaluation import ModelEvaluator, AdvancedEnsemble


class PredictionPipeline:
    """预测管道"""

    def __init__(self, use_downsampled: bool = False):
        """
        初始化预测管道

        Args:
            use_downsampled: 是否使用降采样数据进行训练
        """
        self.use_downsampled = use_downsampled
        self.processor = CryptoDataProcessor(use_downsampled=use_downsampled)
        self.models = {}
        self.ensemble = None
        self.test_predictions = None

        # 提交文件保存路径
        project_root = (
            self.processor.get_project_root()
            if hasattr(self.processor, "get_project_root")
            else self.processor.PROJECT_ROOT
        )
        self.submission_dir = os.path.join(project_root, "submissions")
        os.makedirs(self.submission_dir, exist_ok=True)

    def load_and_preprocess_data(self) -> tuple:
        """
        加载和预处理数据

        Returns:
            (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, test_ids)
        """
        print("=== 数据加载和预处理 ===")

        # 加载训练数据
        train_df = self.processor.load_data("train")
        print(f"训练数据形状: {train_df.shape}")

        # 内存优化
        train_df = self.processor.memory_optimization(train_df)

        # 准备特征
        X, y = self.processor.prepare_features(train_df, is_train=True)

        # 分割数据
        X_train, X_val, y_train, y_val = self.processor.split_data(X, y, test_size=0.2)

        # 特征缩放
        X_train_scaled, X_val_scaled = self.processor.scale_features(
            X_train, X_val, scaler_type="robust"
        )

        # 清理训练数据内存
        del train_df, X, y, X_train, X_val
        gc.collect()

        # 加载测试数据
        print("加载测试数据...")
        test_df = self.processor.load_data("test")
        print(f"测试数据形状: {test_df.shape}")

        # 内存优化
        test_df = self.processor.memory_optimization(test_df)

        # 准备测试特征
        X_test, _ = self.processor.prepare_features(test_df, is_train=False)

        # 获取测试集ID（假设测试集有索引或ID列）
        test_ids = np.arange(1, len(X_test) + 1)  # 从1开始的ID

        # 缩放测试特征
        X_test_scaled = pd.DataFrame(
            self.processor.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

        # 清理测试数据内存
        del test_df, X_test
        gc.collect()

        print(f"预处理完成:")
        print(f"  训练集: {X_train_scaled.shape}")
        print(f"  验证集: {X_val_scaled.shape}")
        print(f"  测试集: {X_test_scaled.shape}")

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, test_ids

    def train_gradient_boosting_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tune_hyperparams: bool = False,
    ) -> Dict[str, Any]:
        """
        训练梯度提升模型

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            tune_hyperparams: 是否进行超参数调优

        Returns:
            训练结果字典
        """
        print("\n=== 训练梯度提升模型 ===")

        results = {}

        # XGBoost
        print("训练XGBoost模型...")
        xgb_trainer = GradientBoostingTrainer("xgboost")
        if tune_hyperparams:
            xgb_results = xgb_trainer.hyperparameter_tuning(
                X_train, y_train, X_val, y_val, n_iter=20
            )
        else:
            xgb_results = xgb_trainer.train(X_train, y_train, X_val, y_val)

        self.models["XGBoost"] = xgb_trainer
        results["XGBoost"] = xgb_results
        xgb_trainer.save_model("xgboost_final.pkl")

        # LightGBM
        print("训练LightGBM模型...")
        lgb_trainer = GradientBoostingTrainer("lightgbm")
        if tune_hyperparams:
            lgb_results = lgb_trainer.hyperparameter_tuning(
                X_train, y_train, X_val, y_val, n_iter=20
            )
        else:
            lgb_results = lgb_trainer.train(X_train, y_train, X_val, y_val)

        self.models["LightGBM"] = lgb_trainer
        results["LightGBM"] = lgb_results
        lgb_trainer.save_model("lightgbm_final.pkl")

        return results

    def train_lstm_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        window_size: int = 10,
    ) -> Dict[str, Any]:
        """
        训练LSTM模型

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            window_size: 时间窗口大小

        Returns:
            训练结果字典
        """
        print(f"\n=== 训练LSTM模型 (窗口大小: {window_size}) ===")

        lstm_trainer = LSTMTrainer(
            window_size=window_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
        )

        lstm_results = lstm_trainer.train(
            X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=10
        )

        self.models["LSTM"] = lstm_trainer

        return lstm_results

    def create_ensemble(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        ensemble_method: str = "weighted_average",
    ) -> Dict[str, Any]:
        """
        创建集成模型

        Args:
            X_val: 验证特征
            y_val: 验证目标
            ensemble_method: 集成方法 ('weighted_average', 'stacking')

        Returns:
            集成结果字典
        """
        print(f"\n=== 创建集成模型 ({ensemble_method}) ===")

        if len(self.models) < 2:
            print("警告: 模型数量少于2个，无法创建集成")
            return {}

        # 创建高级集成
        self.ensemble = AdvancedEnsemble(self.models)

        if ensemble_method == "weighted_average":
            self.ensemble.fit_weighted_average(X_val, y_val, weight_method="pearson")
        elif ensemble_method == "stacking":
            # 注意：堆叠需要额外的训练数据，这里简化处理
            self.ensemble.fit_weighted_average(X_val, y_val, weight_method="pearson")
            print("注意: 使用加权平均代替堆叠（需要更多数据用于堆叠）")

        # 评估集成性能
        ensemble_pred = self.ensemble.predict(X_val)

        from model_evaluation import ModelEvaluator

        evaluator = ModelEvaluator()
        ensemble_metrics = evaluator.calculate_metrics(
            y_val.values, ensemble_pred, "Ensemble"
        )

        print(
            f"集成模型验证集皮尔逊相关系数: {ensemble_metrics['pearson_correlation']:.6f}"
        )

        return ensemble_metrics

    def generate_predictions(
        self, X_test: pd.DataFrame, test_ids: np.ndarray, use_ensemble: bool = True
    ) -> pd.DataFrame:
        """
        生成测试集预测

        Args:
            X_test: 测试特征
            test_ids: 测试集ID
            use_ensemble: 是否使用集成模型

        Returns:
            预测结果DataFrame
        """
        print(f"\n=== 生成测试集预测 (使用集成: {use_ensemble}) ===")

        if use_ensemble and self.ensemble is not None:
            print("使用集成模型进行预测...")
            predictions = self.ensemble.predict(X_test)
        elif len(self.models) > 0:
            print("使用最佳单个模型进行预测...")
            # 选择第一个可用模型
            best_model = list(self.models.values())[0]
            predictions = best_model.predict(X_test)
        else:
            raise ValueError("没有可用的训练模型")

        # 创建提交格式
        submission_df = pd.DataFrame({"ID": test_ids, "prediction": predictions})

        self.test_predictions = submission_df

        print(f"预测完成，预测值统计:")
        print(f"  均值: {predictions.mean():.6f}")
        print(f"  标准差: {predictions.std():.6f}")
        print(f"  最小值: {predictions.min():.6f}")
        print(f"  最大值: {predictions.max():.6f}")

        return submission_df

    def save_submission(
        self, submission_df: pd.DataFrame, filename: Optional[str] = None
    ) -> str:
        """
        保存提交文件

        Args:
            submission_df: 提交数据框
            filename: 文件名，默认使用时间戳

        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{timestamp}.csv"

        filepath = os.path.join(self.submission_dir, filename)
        submission_df.to_csv(filepath, index=False)

        print(f"\n=== 提交文件已保存 ===")
        print(f"文件路径: {filepath}")
        print(f"文件大小: {len(submission_df)} 行")
        print(f"提交格式预览:")
        print(submission_df.head())

        return filepath

    def run_full_pipeline(
        self,
        tune_hyperparams: bool = False,
        use_lstm: bool = True,
        lstm_window_size: int = 10,
        ensemble_method: str = "weighted_average",
    ) -> str:
        """
        运行完整的预测管道

        Args:
            tune_hyperparams: 是否进行超参数调优
            use_lstm: 是否使用LSTM模型
            lstm_window_size: LSTM窗口大小
            ensemble_method: 集成方法

        Returns:
            提交文件路径
        """
        print("开始运行完整预测管道...")

        # 1. 数据预处理
        X_train, X_val, X_test, y_train, y_val, test_ids = (
            self.load_and_preprocess_data()
        )

        # 2. 训练梯度提升模型
        gb_results = self.train_gradient_boosting_models(
            X_train, y_train, X_val, y_val, tune_hyperparams
        )

        # 3. 训练LSTM模型（可选）
        if use_lstm:
            try:
                lstm_results = self.train_lstm_model(
                    X_train, y_train, X_val, y_val, lstm_window_size
                )
            except Exception as e:
                print(f"LSTM训练失败: {e}")
                print("继续使用梯度提升模型...")

        # 4. 创建集成模型
        ensemble_results = self.create_ensemble(X_val, y_val, ensemble_method)

        # 5. 生成预测
        submission_df = self.generate_predictions(X_test, test_ids, use_ensemble=True)

        # 6. 保存提交文件
        submission_path = self.save_submission(submission_df)

        print(f"\n=== 管道运行完成 ===")
        print(f"提交文件: {submission_path}")

        return submission_path


def quick_baseline(use_downsampled: bool = True) -> str:
    """
    快速基线模型

    Args:
        use_downsampled: 是否使用降采样数据

    Returns:
        提交文件路径
    """
    print("运行快速基线模型...")

    pipeline = PredictionPipeline(use_downsampled=use_downsampled)


def medium_scale_training(downsample_ratio: float = 0.3) -> str:
    """
    中等规模训练

    Args:
        downsample_ratio: 降采样比例

    Returns:
        提交文件路径
    """
    print(f"运行中等规模训练 (使用{downsample_ratio*100:.0f}%数据)...")

    # 创建临时降采样数据
    processor = CryptoDataProcessor(
        use_downsampled=True, downsample_ratio=downsample_ratio
    )

    # 强制重新创建降采样数据
    train_df = processor.load_data("train")
    if len(train_df) > 100000:  # 如果数据太大，重新采样
        train_df = train_df.sample(frac=downsample_ratio, random_state=42)
        downsampled_file = os.path.join(
            processor.processed_data_dir,
            f"drw-crypto-train_medium_{int(downsample_ratio*100)}.parquet",
        )
        train_df.to_parquet(downsampled_file)
        print(f"中等规模数据已保存: {downsampled_file}")

    pipeline = PredictionPipeline(use_downsampled=False)  # 不使用默认降采样

    # 简化的管道：只使用梯度提升，不调优超参数
    X_train, X_val, X_test, y_train, y_val, test_ids = (
        pipeline.load_and_preprocess_data()
    )

    # 只训练XGBoost
    xgb_trainer = GradientBoostingTrainer("xgboost")
    xgb_trainer.train(X_train, y_train, X_val, y_val)

    # 预测
    predictions = xgb_trainer.predict(X_test)

    # 创建提交文件
    submission_df = pd.DataFrame({"ID": test_ids, "prediction": predictions})

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_submission_{timestamp}.csv"
    submission_path = pipeline.save_submission(submission_df, filename)

    return submission_path


# 注释掉测试代码，避免导入时执行
# if __name__ == "__main__":
#     # 运行快速基线测试
#     print("测试快速基线模型...")
#     baseline_path = quick_baseline(use_downsampled=True)
#
#     print(f"\n基线模型提交文件: {baseline_path}")
#     print("预测管道测试完成!")
