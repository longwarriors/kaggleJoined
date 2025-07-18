"""
Home Credit Default Risk - 主程序

重构版的完整机器学习流水线主程序。

作者：Augment Agent
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pipeline.main_pipeline import HomeCreditPipeline
from core.utils import get_timestamp


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Home Credit Default Risk Pipeline')
    parser.add_argument('--config-dir', type=str, default=None,
                       help='配置文件目录路径')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'test', 'predict'],
                       help='运行模式: full(完整流水线), test(测试), predict(仅预测)')
    parser.add_argument('--test-data', type=str, default=None,
                       help='测试数据路径（用于预测模式）')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Home Credit Default Risk - 重构版机器学习流水线")
    print("=" * 80)
    print(f"运行模式: {args.mode}")
    print(f"开始时间: {get_timestamp('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # 初始化流水线
        pipeline = HomeCreditPipeline(config_dir=args.config_dir)
        
        if args.mode == 'full':
            # 运行完整流水线
            print("🚀 开始运行完整流水线...")
            results = pipeline.run()
            
            # 显示结果摘要
            print_results_summary(results)
            
            # 生成提交文件
            try:
                submission_file = pipeline.generate_submission()
                print(f"✅ 提交文件已生成: {submission_file}")
            except Exception as e:
                print(f"⚠️  生成提交文件失败: {e}")
        
        elif args.mode == 'test':
            # 测试模式 - 运行简化流水线
            print("🧪 运行测试模式...")
            test_pipeline(pipeline)
        
        elif args.mode == 'predict':
            # 预测模式 - 仅进行预测
            if not args.test_data:
                print("❌ 预测模式需要提供测试数据路径")
                sys.exit(1)
            
            print("🔮 运行预测模式...")
            predict_mode(pipeline, args.test_data)
        
        print("\n" + "=" * 80)
        print("✅ 流水线运行完成")
        print(f"结束时间: {get_timestamp('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断了程序运行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 流水线运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_results_summary(results: dict):
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("📊 流水线结果摘要")
    print("=" * 60)
    
    # 数据信息
    metadata = results.get('pipeline_metadata', {})
    print(f"📈 数据形状: {metadata.get('data_shape', 'N/A')}")
    print(f"🎯 选择特征数: {metadata.get('selected_features_count', 'N/A')}")
    
    # 基线结果
    baseline_results = results.get('baseline_results', {})
    if baseline_results:
        best_baseline = baseline_results.get('best_model', 'N/A')
        baseline_score = baseline_results.get('best_score', 0)
        print(f"📊 最佳基线模型: {best_baseline} (AUC: {baseline_score:.4f})")
    
    # 高级模型结果
    model_results = results.get('model_results', {})
    if model_results:
        best_model = model_results.get('best_model', 'N/A')
        model_score = model_results.get('best_score', 0)
        print(f"🚀 最佳高级模型: {best_model} (AUC: {model_score:.4f})")
    
    # 集成结果
    ensemble_results = results.get('ensemble_results', {})
    if ensemble_results:
        best_ensemble = ensemble_results.get('best_ensemble', 'N/A')
        ensemble_score = ensemble_results.get('best_score', 0)
        print(f"🎭 最佳集成模型: {best_ensemble} (AUC: {ensemble_score:.4f})")
    
    # 最终结果
    final_results = results.get('final_results', {})
    if final_results:
        final_model_type = final_results.get('best_model_type', 'N/A')
        final_model_name = final_results.get('best_model_name', 'N/A')
        final_evaluation = final_results.get('evaluation', {})
        final_auc = final_evaluation.get('roc_auc', 0)
        
        print(f"\n🏆 最终最佳模型:")
        print(f"   类型: {final_model_type}")
        print(f"   名称: {final_model_name}")
        print(f"   AUC: {final_auc:.4f}")
        print(f"   准确率: {final_evaluation.get('accuracy', 0):.4f}")
        print(f"   精确率: {final_evaluation.get('precision', 0):.4f}")
        print(f"   召回率: {final_evaluation.get('recall', 0):.4f}")
        print(f"   F1分数: {final_evaluation.get('f1', 0):.4f}")
        
        # 基线比较
        baseline_comparison = final_results.get('baseline_comparison', {})
        if baseline_comparison:
            improvement = baseline_comparison.get('improvement', 0)
            improvement_pct = baseline_comparison.get('improvement_percentage', 0)
            is_better = baseline_comparison.get('is_better', False)
            
            status = "✅ 超过基线" if is_better else "❌ 未超过基线"
            print(f"\n📈 与基线比较: {status}")
            print(f"   改进幅度: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    print("=" * 60)


def test_pipeline(pipeline):
    """测试流水线"""
    print("运行流水线组件测试...")
    
    try:
        # 创建模拟数据进行测试
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        # 模拟主表数据
        mock_data = {
            'application_train': pd.DataFrame({
                'SK_ID_CURR': range(n_samples),
                'TARGET': np.random.binomial(1, 0.08, n_samples),
                'AMT_INCOME_TOTAL': np.random.lognormal(11, 0.5, n_samples),
                'AMT_CREDIT': np.random.lognormal(12, 0.3, n_samples),
                'AMT_ANNUITY': np.random.lognormal(9, 0.4, n_samples),
                'DAYS_BIRTH': -np.random.randint(18*365, 70*365, n_samples),
                'DAYS_EMPLOYED': -np.random.randint(0, 40*365, n_samples),
                'EXT_SOURCE_1': np.random.uniform(0, 1, n_samples),
                'EXT_SOURCE_2': np.random.uniform(0, 1, n_samples),
                'EXT_SOURCE_3': np.random.uniform(0, 1, n_samples),
            })
        }
        
        # 设置模拟数据
        pipeline.raw_data = mock_data
        pipeline.cleaned_data = mock_data
        
        # 测试特征工程
        print("✅ 测试特征工程...")
        pipeline._build_features()
        
        # 测试特征选择
        print("✅ 测试特征选择...")
        pipeline._select_features()
        
        # 测试基线模型
        print("✅ 测试基线模型...")
        baseline_results = pipeline._train_baseline_models()
        
        print("🎉 测试完成！所有组件运行正常。")
        
        # 显示测试结果
        if baseline_results:
            best_model = baseline_results.get('best_model', 'N/A')
            best_score = baseline_results.get('best_score', 0)
            print(f"📊 测试结果 - 最佳基线模型: {best_model} (AUC: {best_score:.4f})")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


def predict_mode(pipeline, test_data_path):
    """预测模式"""
    print(f"加载测试数据: {test_data_path}")
    
    # 这里需要先训练模型或加载已训练的模型
    print("⚠️  预测模式需要先运行完整流水线或加载已训练的模型")
    print("请先使用 --mode full 运行完整流水线")


if __name__ == "__main__":
    main()
