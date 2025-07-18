"""
模型评估模块

提供全面的模型评估功能，包括各种评估指标、交叉验证、可视化等。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseEvaluator
from ..core.utils import get_timestamp


class ModelEvaluator(BaseEvaluator):
    """
    模型评估器
    
    提供全面的模型评估功能
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化模型评估器
        
        Args:
            config: 评估配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.plots_dir = Path("outputs/evaluation_plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_pred_proba: Optional[np.ndarray] = None, **kwargs) -> Dict:
        """
        评估预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            y_pred_proba: 预测概率
            **kwargs: 其他参数
            
        Returns:
            评估结果字典
        """
        self._log_info("开始模型评估")
        
        results = {}
        
        # 基本分类指标
        results.update(self._calculate_classification_metrics(y_true, y_pred, y_pred_proba))
        
        # 混淆矩阵
        results['confusion_matrix'] = self._calculate_confusion_matrix(y_true, y_pred)
        
        # 分类报告
        results['classification_report'] = self._generate_classification_report(y_true, y_pred)
        
        # ROC和PR曲线数据
        if y_pred_proba is not None:
            results['roc_data'] = self._calculate_roc_data(y_true, y_pred_proba)
            results['pr_data'] = self._calculate_pr_data(y_true, y_pred_proba)
        
        self.evaluation_results = results
        self._log_info("模型评估完成")
        
        return results
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                           cv_config: Optional[Dict] = None) -> Dict:
        """
        交叉验证评估模型
        
        Args:
            model: 模型实例
            X: 特征数据
            y: 目标变量
            cv_config: 交叉验证配置
            
        Returns:
            交叉验证结果
        """
        if cv_config is None:
            cv_config = self.config.get('evaluation', {}).get('cross_validation', {})
        
        # 设置交叉验证
        cv = StratifiedKFold(
            n_splits=cv_config.get('n_splits', 5),
            shuffle=cv_config.get('shuffle', True),
            random_state=cv_config.get('random_state', 42)
        )
        
        # 获取评估指标
        metrics = self.config.get('evaluation', {}).get('metrics', ['roc_auc'])
        
        cv_results = {}
        
        for metric in metrics:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                cv_results[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist(),
                    'min': scores.min(),
                    'max': scores.max()
                }
                self._log_info(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                self._log_warning(f"计算 {metric} 失败: {e}")
                cv_results[metric] = {
                    'mean': 0.0, 'std': 0.0, 'scores': [], 'min': 0.0, 'max': 0.0
                }
        
        return cv_results
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """计算分类指标"""
        metrics = {}
        
        try:
            # 基本指标
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            
            # 如果有概率预测，计算AUC和log loss
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
        except Exception as e:
            self._log_error(f"计算分类指标失败: {e}")
            metrics = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                'f1': 0.0, 'roc_auc': 0.0, 'log_loss': 0.0
            }
        
        return metrics
    
    def _calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算混淆矩阵"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            return {
                'matrix': cm.tolist(),
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1])
            }
        except Exception as e:
            self._log_error(f"计算混淆矩阵失败: {e}")
            return {'matrix': [[0, 0], [0, 0]], 'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    def _generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """生成分类报告"""
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
            return report
        except Exception as e:
            self._log_error(f"生成分类报告失败: {e}")
            return {}
    
    def _calculate_roc_data(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """计算ROC曲线数据"""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            auc = roc_auc_score(y_true, y_pred_proba)
            
            return {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': auc
            }
        except Exception as e:
            self._log_error(f"计算ROC数据失败: {e}")
            return {'fpr': [], 'tpr': [], 'thresholds': [], 'auc': 0.0}
    
    def _calculate_pr_data(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """计算PR曲线数据"""
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            
            return {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist()
            }
        except Exception as e:
            self._log_error(f"计算PR数据失败: {e}")
            return {'precision': [], 'recall': [], 'thresholds': []}
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            save_plot: bool = True, model_name: str = "Model") -> Optional[str]:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            save_plot: 是否保存图表
            model_name: 模型名称
            
        Returns:
            图表文件路径（如果保存）
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            if save_plot:
                plot_path = self.plots_dir / f"confusion_matrix_{model_name}_{get_timestamp()}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self._log_info(f"混淆矩阵图已保存: {plot_path}")
                return str(plot_path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            self._log_error(f"绘制混淆矩阵失败: {e}")
            return None
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_plot: bool = True, model_name: str = "Model") -> Optional[str]:
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实值
            y_pred_proba: 预测概率
            save_plot: 是否保存图表
            model_name: 模型名称
            
        Returns:
            图表文件路径（如果保存）
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = roc_auc_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if save_plot:
                plot_path = self.plots_dir / f"roc_curve_{model_name}_{get_timestamp()}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self._log_info(f"ROC曲线图已保存: {plot_path}")
                return str(plot_path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            self._log_error(f"绘制ROC曲线失败: {e}")
            return None
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_plot: bool = True, model_name: str = "Model") -> Optional[str]:
        """
        绘制PR曲线
        
        Args:
            y_true: 真实值
            y_pred_proba: 预测概率
            save_plot: 是否保存图表
            model_name: 模型名称
            
        Returns:
            图表文件路径（如果保存）
        """
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2, label=f'{model_name}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_plot:
                plot_path = self.plots_dir / f"pr_curve_{model_name}_{get_timestamp()}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self._log_info(f"PR曲线图已保存: {plot_path}")
                return str(plot_path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            self._log_error(f"绘制PR曲线失败: {e}")
            return None
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            model_results: 模型结果字典，格式为 {model_name: evaluation_results}
            
        Returns:
            比较结果
        """
        if not model_results:
            return {}
        
        # 获取主要指标
        primary_metric = self.config.get('evaluation', {}).get('primary_metric', 'roc_auc')
        
        comparison = {
            'models': list(model_results.keys()),
            'primary_metric': primary_metric,
            'scores': {},
            'rankings': {},
            'best_model': None,
            'best_score': 0.0
        }
        
        # 收集各模型的分数
        for model_name, results in model_results.items():
            score = results.get(primary_metric, 0.0)
            comparison['scores'][model_name] = score
            
            if score > comparison['best_score']:
                comparison['best_score'] = score
                comparison['best_model'] = model_name
        
        # 排名
        sorted_models = sorted(comparison['scores'].items(), key=lambda x: x[1], reverse=True)
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            comparison['rankings'][model_name] = rank
        
        self._log_info(f"模型比较完成，最佳模型: {comparison['best_model']} ({primary_metric}: {comparison['best_score']:.4f})")
        
        return comparison
    
    def generate_evaluation_report(self, model_results: Dict[str, Any], 
                                 model_name: str = "Model") -> str:
        """
        生成评估报告
        
        Args:
            model_results: 模型评估结果
            model_name: 模型名称
            
        Returns:
            报告内容
        """
        report = f"# {model_name} 评估报告\n\n"
        report += f"生成时间: {get_timestamp('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 主要指标
        report += "## 主要指标\n\n"
        for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']:
            if metric in model_results:
                report += f"- {metric.upper()}: {model_results[metric]:.4f}\n"
        
        # 混淆矩阵
        if 'confusion_matrix' in model_results:
            cm = model_results['confusion_matrix']
            report += "\n## 混淆矩阵\n\n"
            report += f"- True Negatives: {cm['tn']}\n"
            report += f"- False Positives: {cm['fp']}\n"
            report += f"- False Negatives: {cm['fn']}\n"
            report += f"- True Positives: {cm['tp']}\n"
        
        # 分类报告
        if 'classification_report' in model_results:
            report += "\n## 详细分类报告\n\n"
            class_report = model_results['classification_report']
            if '0' in class_report and '1' in class_report:
                report += "### 类别 0 (负类)\n"
                report += f"- Precision: {class_report['0']['precision']:.4f}\n"
                report += f"- Recall: {class_report['0']['recall']:.4f}\n"
                report += f"- F1-score: {class_report['0']['f1-score']:.4f}\n\n"
                
                report += "### 类别 1 (正类)\n"
                report += f"- Precision: {class_report['1']['precision']:.4f}\n"
                report += f"- Recall: {class_report['1']['recall']:.4f}\n"
                report += f"- F1-score: {class_report['1']['f1-score']:.4f}\n"
        
        return report
