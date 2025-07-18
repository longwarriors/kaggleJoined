"""
推理管线使用示例

展示如何使用训练好的模型进行推理预测。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.inference_pipeline import InferencePipeline
from models.serializer import ModelSerializer


def demonstrate_model_saving():
    """演示模型保存功能"""
    print("🔧 模型保存功能演示")
    print("=" * 50)
    
    # 创建模型序列化器
    serializer = ModelSerializer()
    
    # 列出可用的模型
    available_models = serializer.list_available_models()
    
    print(f"📊 可用模型数量: {len(available_models)}")
    
    for i, model_info in enumerate(available_models[:5]):  # 显示前5个
        print(f"  {i+1}. {model_info['model_name']} ({model_info['model_type']})")
        print(f"     版本: {model_info['version']}")
        print(f"     时间: {model_info['timestamp']}")
        print(f"     存在: {'✅' if model_info['exists'] else '❌'}")
        print()


def demonstrate_inference_pipeline():
    """演示推理管线功能"""
    print("🚀 推理管线功能演示")
    print("=" * 50)
    
    try:
        # 1. 初始化推理管线
        print("1️⃣ 初始化推理管线...")
        inference_pipeline = InferencePipeline()
        
        # 2. 加载模型
        print("2️⃣ 加载训练好的模型...")
        models = inference_pipeline.load_models(latest=True)
        
        if not models:
            print("❌ 没有找到可用的模型")
            return
        
        print(f"✅ 成功加载 {len(models)} 个模型:")
        for model_name in models.keys():
            print(f"  - {model_name}")
        
        # 3. 加载特征工程器
        print("3️⃣ 加载特征工程器...")
        feature_builder = inference_pipeline.load_feature_builder()
        print("✅ 特征工程器加载完成")
        
        # 4. 获取模型信息
        print("4️⃣ 模型信息:")
        model_info = inference_pipeline.get_model_info()
        for model_name, info in model_info.items():
            print(f"  {model_name}:")
            print(f"    类型: {info['type']}")
            print(f"    版本: {info['version']}")
            print(f"    特征数: {info['feature_count']}")
            print(f"    性能: {info['performance']}")
        
        return inference_pipeline
        
    except Exception as e:
        print(f"❌ 推理管线初始化失败: {e}")
        return None


def demonstrate_single_prediction(inference_pipeline):
    """演示单样本预测"""
    if inference_pipeline is None:
        print("❌ 推理管线未初始化")
        return
    
    print("\n🎯 单样本预测演示")
    print("=" * 50)
    
    # 创建示例数据
    sample_data = {
        'SK_ID_CURR': 100001,
        'AMT_INCOME_TOTAL': 202500.0,
        'AMT_CREDIT': 406597.5,
        'AMT_ANNUITY': 24700.5,
        'AMT_GOODS_PRICE': 351000.0,
        'NAME_CONTRACT_TYPE': 'Cash loans',
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'Y',
        'CNT_CHILDREN': 0,
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Single / not married',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'DAYS_BIRTH': -9461,
        'DAYS_EMPLOYED': -637,
        'REGION_RATING_CLIENT': 2,
        'REGION_RATING_CLIENT_W_CITY': 2,
        'EXT_SOURCE_1': 0.083037,
        'EXT_SOURCE_2': 0.262949,
        'EXT_SOURCE_3': 0.139376
    }
    
    try:
        # 进行预测
        print("🔮 开始预测...")
        result = inference_pipeline.predict_single(sample_data)
        
        print("✅ 预测完成:")
        print(f"  预测结果: {result['prediction']:.4f}")
        if result['probability'] is not None:
            print(f"  违约概率: {result['probability']:.4f}")
        
        print("  各模型预测:")
        for model_name, pred in result['individual_predictions'].items():
            print(f"    {model_name}: {pred:.4f}")
        
    except Exception as e:
        print(f"❌ 单样本预测失败: {e}")


def demonstrate_batch_prediction(inference_pipeline):
    """演示批量预测"""
    if inference_pipeline is None:
        print("❌ 推理管线未初始化")
        return
    
    print("\n📊 批量预测演示")
    print("=" * 50)
    
    # 检查测试数据是否存在
    test_data_path = "../../data/raw/home-credit-default-risk/application_test.csv"
    
    if not Path(test_data_path).exists():
        print(f"❌ 测试数据文件不存在: {test_data_path}")
        print("💡 创建模拟测试数据...")
        
        # 创建模拟数据
        np.random.seed(42)
        n_samples = 100
        
        mock_data = pd.DataFrame({
            'SK_ID_CURR': range(100001, 100001 + n_samples),
            'AMT_INCOME_TOTAL': np.random.normal(200000, 50000, n_samples),
            'AMT_CREDIT': np.random.normal(400000, 100000, n_samples),
            'AMT_ANNUITY': np.random.normal(25000, 5000, n_samples),
            'AMT_GOODS_PRICE': np.random.normal(350000, 80000, n_samples),
            'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], n_samples),
            'CODE_GENDER': np.random.choice(['M', 'F'], n_samples),
            'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_samples),
            'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n_samples),
            'CNT_CHILDREN': np.random.randint(0, 5, n_samples),
            'DAYS_BIRTH': np.random.randint(-25000, -6000, n_samples),
            'DAYS_EMPLOYED': np.random.randint(-15000, 0, n_samples),
            'EXT_SOURCE_1': np.random.uniform(0, 1, n_samples),
            'EXT_SOURCE_2': np.random.uniform(0, 1, n_samples),
            'EXT_SOURCE_3': np.random.uniform(0, 1, n_samples)
        })
        
        # 保存模拟数据
        mock_data_path = "./outputs/mock_test_data.csv"
        mock_data.to_csv(mock_data_path, index=False)
        test_data_path = mock_data_path
        print(f"✅ 模拟数据已创建: {mock_data_path}")
    
    try:
        # 批量预测
        print("🔮 开始批量预测...")
        output_path = "./outputs/batch_predictions.csv"
        
        batch_results = inference_pipeline.batch_predict(
            input_file=test_data_path,
            output_file=output_path,
            batch_size=50
        )
        
        print("✅ 批量预测完成:")
        print(f"  处理样本数: {batch_results['processed_samples']}")
        print(f"  输出文件: {batch_results['output_file']}")
        
        # 显示推理统计
        stats = batch_results['inference_stats']
        print(f"  总预测次数: {stats['total_predictions']}")
        print(f"  总耗时: {stats['total_time']:.2f}秒")
        print(f"  平均每样本耗时: {stats['avg_time_per_prediction']:.4f}秒")
        
        # 显示预测结果样例
        if Path(output_path).exists():
            results_df = pd.read_csv(output_path)
            print(f"\n📋 预测结果样例 (前5行):")
            print(results_df.head())
        
    except Exception as e:
        print(f"❌ 批量预测失败: {e}")


def demonstrate_model_management():
    """演示模型管理功能"""
    print("\n🗂️ 模型管理功能演示")
    print("=" * 50)
    
    serializer = ModelSerializer()
    
    # 1. 列出所有模型
    print("1️⃣ 所有可用模型:")
    all_models = serializer.list_available_models()
    
    for model in all_models:
        print(f"  📦 {model['model_name']} ({model['model_type']})")
        print(f"     版本: {model['version']}")
        print(f"     时间: {model['timestamp']}")
        print(f"     存在: {'✅' if model['exists'] else '❌'}")
        
        if 'performance' in model:
            perf = model['performance']
            if 'cv_auc' in perf:
                print(f"     性能: AUC {perf['cv_auc']:.4f}")
        print()
    
    # 2. 按类型筛选
    print("2️⃣ LightGBM模型:")
    lgb_models = serializer.list_available_models(model_type='lightgbm')
    for model in lgb_models:
        print(f"  🌟 {model['model_name']} - {model['version']}")
    
    # 3. 按名称筛选
    print("3️⃣ 集成模型:")
    ensemble_models = serializer.list_available_models(model_type='ensemble')
    for model in ensemble_models:
        print(f"  🎯 {model['model_name']} - {model['version']}")


def main():
    """主函数"""
    print("🎉 Home Credit 推理系统演示")
    print("=" * 60)
    
    # 1. 模型保存功能演示
    demonstrate_model_saving()
    
    # 2. 推理管线演示
    inference_pipeline = demonstrate_inference_pipeline()
    
    if inference_pipeline is not None:
        # 3. 单样本预测演示
        demonstrate_single_prediction(inference_pipeline)
        
        # 4. 批量预测演示
        demonstrate_batch_prediction(inference_pipeline)
    
    # 5. 模型管理演示
    demonstrate_model_management()
    
    print("\n🎊 演示完成！")
    print("\n💡 使用提示:")
    print("  1. 训练完成后，模型会自动保存到 outputs/models/ 目录")
    print("  2. 使用 InferencePipeline 可以加载模型进行推理")
    print("  3. 支持单样本预测和批量预测")
    print("  4. 模型元数据保存在 outputs/metadata/ 目录")
    print("  5. 可以通过 ModelSerializer 管理模型版本")


if __name__ == "__main__":
    main()
