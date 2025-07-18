"""
生产环境推理脚本

用于加载训练好的模型进行预测的简化脚本。

使用方法:
    python predict.py --input data.csv --output predictions.csv
    python predict.py --input data.csv --output predictions.csv --models lgb_1,lgb_2
    python predict.py --single --data '{"AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 400000}'

作者：Augment Agent
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from pipeline.inference_pipeline import InferencePipeline


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Home Credit Default Risk 预测工具')
    
    # 输入输出参数
    parser.add_argument('--input', '-i', type=str, help='输入数据文件路径 (CSV格式)')
    parser.add_argument('--output', '-o', type=str, help='输出预测文件路径 (CSV格式)')
    
    # 单样本预测参数
    parser.add_argument('--single', action='store_true', help='单样本预测模式')
    parser.add_argument('--data', type=str, help='单样本数据 (JSON格式)')
    
    # 模型参数
    parser.add_argument('--models', type=str, help='指定使用的模型名称 (逗号分隔)')
    parser.add_argument('--ensemble', type=str, default='average', 
                       choices=['average', 'weighted', 'voting'],
                       help='集成方法')
    
    # 其他参数
    parser.add_argument('--batch-size', type=int, default=1000, help='批处理大小')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    return parser.parse_args()


def setup_inference_pipeline(config_path=None, verbose=False):
    """设置推理管线"""
    try:
        if verbose:
            print("🚀 初始化推理管线...")
        
        # 初始化推理管线
        if config_path:
            # 加载自定义配置
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            pipeline = InferencePipeline(config)
        else:
            pipeline = InferencePipeline()
        
        # 加载模型
        if verbose:
            print("📦 加载训练好的模型...")
        
        models = pipeline.load_models(latest=True)
        
        if not models:
            print("❌ 错误: 没有找到可用的模型")
            print("💡 请先运行训练脚本生成模型")
            return None
        
        if verbose:
            print(f"✅ 成功加载 {len(models)} 个模型:")
            for model_name in models.keys():
                print(f"  - {model_name}")
        
        # 加载特征工程器
        if verbose:
            print("🔧 加载特征工程器...")
        
        pipeline.load_feature_builder()
        
        if verbose:
            print("✅ 推理管线准备完成")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ 推理管线初始化失败: {e}")
        return None


def single_prediction(pipeline, data_str, verbose=False):
    """单样本预测"""
    try:
        # 解析JSON数据
        data = json.loads(data_str)
        
        if verbose:
            print("🎯 开始单样本预测...")
            print(f"📊 输入数据: {len(data)} 个特征")
        
        # 进行预测
        result = pipeline.predict_single(data)
        
        # 输出结果
        print("✅ 预测完成:")
        print(f"  违约概率: {result['prediction']:.4f}")
        
        if result['probability'] is not None:
            print(f"  置信度: {result['probability']:.4f}")
        
        if verbose:
            print("  各模型预测:")
            for model_name, pred in result['individual_predictions'].items():
                print(f"    {model_name}: {pred:.4f}")
        
        return result
        
    except json.JSONDecodeError:
        print("❌ 错误: 无法解析JSON数据")
        print("💡 示例格式: '{\"AMT_INCOME_TOTAL\": 200000, \"AMT_CREDIT\": 400000}'")
        return None
    except Exception as e:
        print(f"❌ 单样本预测失败: {e}")
        return None


def batch_prediction(pipeline, input_file, output_file, batch_size=1000, 
                    model_names=None, ensemble_method='average', verbose=False):
    """批量预测"""
    try:
        # 检查输入文件
        if not Path(input_file).exists():
            print(f"❌ 错误: 输入文件不存在: {input_file}")
            return False
        
        if verbose:
            print(f"📊 开始批量预测...")
            print(f"  输入文件: {input_file}")
            print(f"  输出文件: {output_file}")
            print(f"  批处理大小: {batch_size}")
            if model_names:
                print(f"  指定模型: {model_names}")
            print(f"  集成方法: {ensemble_method}")
        
        # 执行批量预测
        results = pipeline.batch_predict(
            input_file=input_file,
            output_file=output_file,
            batch_size=batch_size,
            model_names=model_names
        )
        
        print("✅ 批量预测完成:")
        print(f"  处理样本数: {results['processed_samples']}")
        print(f"  输出文件: {results['output_file']}")
        
        # 显示性能统计
        stats = results['inference_stats']
        print(f"  总耗时: {stats['total_time']:.2f}秒")
        print(f"  平均每样本耗时: {stats['avg_time_per_prediction']:.4f}秒")
        
        # 显示预测结果样例
        if verbose and Path(output_file).exists():
            try:
                results_df = pd.read_csv(output_file)
                print(f"\n📋 预测结果样例:")
                print(results_df.head())
                
                # 统计信息
                if 'probability' in results_df.columns:
                    prob_stats = results_df['probability'].describe()
                    print(f"\n📈 违约概率统计:")
                    print(f"  平均值: {prob_stats['mean']:.4f}")
                    print(f"  中位数: {prob_stats['50%']:.4f}")
                    print(f"  标准差: {prob_stats['std']:.4f}")
                    print(f"  最小值: {prob_stats['min']:.4f}")
                    print(f"  最大值: {prob_stats['max']:.4f}")
                    
                    # 风险分布
                    high_risk = (results_df['probability'] > 0.5).sum()
                    total = len(results_df)
                    print(f"\n⚠️ 风险分布:")
                    print(f"  高风险客户: {high_risk} ({high_risk/total*100:.1f}%)")
                    print(f"  低风险客户: {total-high_risk} ({(total-high_risk)/total*100:.1f}%)")
                    
            except Exception as e:
                if verbose:
                    print(f"⚠️ 无法显示结果样例: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量预测失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_arguments()
    
    # 验证参数
    if not args.single and not args.input:
        print("❌ 错误: 必须指定输入文件 (--input) 或使用单样本模式 (--single)")
        sys.exit(1)
    
    if args.single and not args.data:
        print("❌ 错误: 单样本模式需要提供数据 (--data)")
        sys.exit(1)
    
    if not args.single and not args.output:
        print("❌ 错误: 批量预测模式需要指定输出文件 (--output)")
        sys.exit(1)
    
    # 设置推理管线
    pipeline = setup_inference_pipeline(args.config, args.verbose)
    if pipeline is None:
        sys.exit(1)
    
    # 解析模型名称
    model_names = None
    if args.models:
        model_names = [name.strip() for name in args.models.split(',')]
        if args.verbose:
            print(f"🎯 指定使用模型: {model_names}")
    
    # 执行预测
    if args.single:
        # 单样本预测
        result = single_prediction(pipeline, args.data, args.verbose)
        if result is None:
            sys.exit(1)
    else:
        # 批量预测
        success = batch_prediction(
            pipeline=pipeline,
            input_file=args.input,
            output_file=args.output,
            batch_size=args.batch_size,
            model_names=model_names,
            ensemble_method=args.ensemble,
            verbose=args.verbose
        )
        if not success:
            sys.exit(1)
    
    print("🎉 预测完成!")


if __name__ == "__main__":
    main()
