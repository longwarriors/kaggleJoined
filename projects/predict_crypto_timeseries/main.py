"""
DRW 加密市场预测比赛 - 主运行脚本
完整的机器学习管道：数据预处理、模型训练、评估、集成和预测
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_submission import PredictionPipeline, quick_baseline
from data_preprocessing import CryptoDataProcessor
from model_evaluation import ModelEvaluator

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DRW 加密市场预测比赛')
    
    # 数据相关参数
    parser.add_argument('--use_downsampled', action='store_true', default=False,
                       help='使用降采样数据进行快速测试')
    parser.add_argument('--downsample_ratio', type=float, default=0.1,
                       help='降采样比例 (默认: 0.1)')
    
    # 模型训练参数
    parser.add_argument('--tune_hyperparams', action='store_true', default=False,
                       help='进行超参数调优')
    parser.add_argument('--use_lstm', action='store_true', default=True,
                       help='使用LSTM模型')
    parser.add_argument('--lstm_window_size', type=int, default=10,
                       help='LSTM时间窗口大小 (默认: 10)')
    
    # 集成参数
    parser.add_argument('--ensemble_method', type=str, default='weighted_average',
                       choices=['weighted_average', 'stacking'],
                       help='集成方法 (默认: weighted_average)')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='full',
                       choices=['baseline', 'full', 'data_analysis'],
                       help='运行模式: baseline(快速基线), full(完整管道), data_analysis(数据分析)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: submissions/)')
    
    args = parser.parse_args()
    
    print("=== DRW 加密市场预测比赛 ===")
    print(f"运行模式: {args.mode}")
    print(f"使用降采样: {args.use_downsampled}")
    if args.use_downsampled:
        print(f"降采样比例: {args.downsample_ratio}")
    print(f"超参数调优: {args.tune_hyperparams}")
    print(f"使用LSTM: {args.use_lstm}")
    print(f"集成方法: {args.ensemble_method}")
    print()
    
    if args.mode == 'data_analysis':
        # 数据分析模式
        run_data_analysis(args.use_downsampled, args.downsample_ratio)
        
    elif args.mode == 'baseline':
        # 快速基线模式
        print("运行快速基线模型...")
        submission_path = quick_baseline(use_downsampled=args.use_downsampled)
        print(f"\n✅ 基线模型完成!")
        print(f"提交文件: {submission_path}")
        
    elif args.mode == 'full':
        # 完整管道模式
        print("运行完整预测管道...")
        
        pipeline = PredictionPipeline(use_downsampled=args.use_downsampled)
        
        submission_path = pipeline.run_full_pipeline(
            tune_hyperparams=args.tune_hyperparams,
            use_lstm=args.use_lstm,
            lstm_window_size=args.lstm_window_size,
            ensemble_method=args.ensemble_method
        )
        
        print(f"\n✅ 完整管道完成!")
        print(f"提交文件: {submission_path}")
    
    else:
        print(f"未知的运行模式: {args.mode}")
        return


def run_data_analysis(use_downsampled: bool = True, downsample_ratio: float = 0.1):
    """运行数据分析"""
    print("=== 数据分析 ===")
    
    # 初始化数据处理器
    processor = CryptoDataProcessor(use_downsampled=use_downsampled, 
                                   downsample_ratio=downsample_ratio)
    
    # 加载训练数据
    print("加载训练数据...")
    train_df = processor.load_data('train')
    
    # 数据分析
    analysis = processor.analyze_data(train_df)
    
    print("\n=== 训练数据分析结果 ===")
    print(f"数据形状: {analysis['shape']}")
    print(f"内存使用: {analysis['memory_usage_mb']:.2f} MB")
    print(f"数值列数量: {analysis['numeric_columns']}")
    print(f"分类列数量: {analysis['categorical_columns']}")
    print(f"缺失值总数: {analysis['missing_values']}")
    
    if 'target_stats' in analysis:
        print(f"\n目标变量统计:")
        for key, value in analysis['target_stats'].items():
            print(f"  {key}: {value:.6f}")
    
    # 特征分析
    print(f"\n=== 特征信息 ===")
    print(f"总特征数: {len(train_df.columns) - 1}")  # 减去目标变量
    
    # 交易量特征
    volume_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    print(f"交易量特征: {volume_features}")
    
    # X特征
    x_features = [col for col in train_df.columns if col.startswith('X')]
    print(f"X特征数量: {len(x_features)}")
    print(f"X特征范围: {x_features[0]} 到 {x_features[-1]}")
    
    # 数据类型分析
    print(f"\n=== 数据类型分析 ===")
    dtype_counts = train_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"{dtype}: {count} 列")
    
    # 基本统计
    print(f"\n=== 基本统计信息 ===")
    numeric_cols = train_df.select_dtypes(include=['number']).columns
    stats = train_df[numeric_cols].describe()
    print("前5个数值列统计:")
    print(stats.iloc[:, :5].round(4))
    
    # 目标变量分布
    if 'label' in train_df.columns:
        target = train_df['label']
        print(f"\n=== 目标变量分布 ===")
        print(f"均值: {target.mean():.6f}")
        print(f"标准差: {target.std():.6f}")
        print(f"最小值: {target.min():.6f}")
        print(f"最大值: {target.max():.6f}")
        print(f"偏度: {target.skew():.6f}")
        print(f"峰度: {target.kurtosis():.6f}")
    
    # 加载测试数据信息
    print(f"\n=== 测试数据信息 ===")
    test_df = processor.load_data('test')
    test_analysis = processor.analyze_data(test_df)
    print(f"测试集形状: {test_analysis['shape']}")
    print(f"测试集内存使用: {test_analysis['memory_usage_mb']:.2f} MB")
    
    # 提交格式
    print(f"\n=== 提交格式 ===")
    submission_df = processor.load_data('submission')
    print(f"提交文件形状: {submission_df.shape}")
    print(f"提交文件列: {list(submission_df.columns)}")
    print("提交文件示例:")
    print(submission_df.head())
    
    print(f"\n✅ 数据分析完成!")


def run_quick_test():
    """运行快速测试"""
    print("=== 快速功能测试 ===")
    
    try:
        # 测试数据加载
        processor = CryptoDataProcessor(use_downsampled=True, downsample_ratio=0.01)
        train_df = processor.load_data('train')
        print(f"✅ 数据加载成功: {train_df.shape}")
        
        # 测试数据预处理
        X, y = processor.prepare_features(train_df)
        X_train, X_val, y_train, y_val = processor.split_data(X, y)
        X_train_scaled, X_val_scaled = processor.scale_features(X_train, X_val)
        print(f"✅ 数据预处理成功: 训练集{X_train_scaled.shape}, 验证集{X_val_scaled.shape}")
        
        # 测试模型训练
        from gradient_boosting_models import GradientBoostingTrainer
        trainer = GradientBoostingTrainer('xgboost')
        results = trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
        print(f"✅ 模型训练成功: 验证集皮尔逊相关系数 {results['val_pearson']:.4f}")
        
        print("✅ 所有功能测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果没有命令行参数，运行快速测试
    if len(sys.argv) == 1:
        print("未提供命令行参数，运行快速测试...")
        run_quick_test()
        print("\n使用示例:")
        print("python main.py --mode baseline --use_downsampled  # 快速基线")
        print("python main.py --mode full --tune_hyperparams     # 完整管道")
        print("python main.py --mode data_analysis               # 数据分析")
    else:
        main()
