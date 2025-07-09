"""
极简稳定训练 - 回到最基础的方法
基于0.07643成功经验，避免任何复杂特征工程
目标: 稳定提升，避免过拟合
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from scipy.stats import pearsonr
import time
import os

def select_raw_features_only(X, y, n_features=15):
    """只选择原始特征，不做任何工程"""
    print(f"🎯 只选择原始的{n_features}个特征...")
    
    # 计算相关性
    correlations = {}
    for col in X.columns:
        try:
            x_col = X[col].fillna(X[col].median())
            y_clean = y.fillna(y.median())
            corr, _ = pearsonr(x_col, y_clean)
            if not np.isnan(corr):
                correlations[col] = abs(corr)
            else:
                correlations[col] = 0
        except:
            correlations[col] = 0
    
    # 选择最重要的原始特征
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    selected_features = [col for col, _ in sorted_features[:n_features]]
    
    print(f"最高相关性: {sorted_features[0][1]:.6f}")
    print(f"选择的特征: {selected_features}")
    
    return X[selected_features].copy(), selected_features

def train_ultra_simple_models(X_train, y_train, X_val, y_val):
    """训练极简模型"""
    print("🤖 训练极简模型...")
    
    results = {}
    
    # 1. Ridge回归 (多个alpha)
    ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_ridge_corr = -1
    best_ridge_alpha = None
    
    for alpha in ridge_alphas:
        ridge_model = Ridge(alpha=alpha, random_state=42)
        ridge_model.fit(X_train, y_train)
        
        ridge_pred = ridge_model.predict(X_val)
        ridge_corr, _ = pearsonr(y_val, ridge_pred)
        
        results[f'ridge_{alpha}'] = {
            'model': ridge_model, 
            'val_corr': ridge_corr, 
            'predictions': ridge_pred
        }
        
        if ridge_corr > best_ridge_corr:
            best_ridge_corr = ridge_corr
            best_ridge_alpha = alpha
        
        print(f"  Ridge (α={alpha:5.3f}): {ridge_corr:.4f}")
    
    print(f"🏆 最佳Ridge: α={best_ridge_alpha}, 相关系数={best_ridge_corr:.4f}")
    
    # 2. Lasso回归
    lasso_alphas = [0.001, 0.01, 0.1]
    best_lasso_corr = -1
    
    for alpha in lasso_alphas:
        try:
            lasso_model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
            lasso_model.fit(X_train, y_train)
            
            lasso_pred = lasso_model.predict(X_val)
            lasso_corr, _ = pearsonr(y_val, lasso_pred)
            
            results[f'lasso_{alpha}'] = {
                'model': lasso_model, 
                'val_corr': lasso_corr, 
                'predictions': lasso_pred
            }
            
            if lasso_corr > best_lasso_corr:
                best_lasso_corr = lasso_corr
            
            print(f"  Lasso (α={alpha:5.3f}): {lasso_corr:.4f}")
        except:
            print(f"  Lasso (α={alpha:5.3f}): 失败")
    
    # 3. ElasticNet
    elastic_params = [(0.01, 0.5), (0.1, 0.5)]
    for alpha, l1_ratio in elastic_params:
        try:
            elastic_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
            elastic_model.fit(X_train, y_train)
            
            elastic_pred = elastic_model.predict(X_val)
            elastic_corr, _ = pearsonr(y_val, elastic_pred)
            
            results[f'elastic_{alpha}_{l1_ratio}'] = {
                'model': elastic_model, 
                'val_corr': elastic_corr, 
                'predictions': elastic_pred
            }
            print(f"  ElasticNet (α={alpha:4.2f}, l1={l1_ratio:3.1f}): {elastic_corr:.4f}")
        except:
            print(f"  ElasticNet (α={alpha:4.2f}, l1={l1_ratio:3.1f}): 失败")
    
    # 4. 选择最佳模型
    valid_results = {k: v for k, v in results.items() 
                    if not np.isnan(v['val_corr']) and v['val_corr'] > 0}
    
    if len(valid_results) == 0:
        print("❌ 没有有效的模型")
        return results
    
    best_model_name = max(valid_results.keys(), key=lambda k: valid_results[k]['val_corr'])
    best_model = valid_results[best_model_name]
    
    print(f"\n🏆 最佳模型: {best_model_name}")
    print(f"🎯 最佳验证相关系数: {best_model['val_corr']:.4f}")
    
    # 5. 简单集成 (只用前3个最好的)
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['val_corr'], reverse=True)
    top3_models = dict(sorted_models[:3])
    
    print(f"Top-3模型: {list(top3_models.keys())}")
    
    # 简单平均集成
    ensemble_pred = np.zeros_like(y_val)
    for model_info in top3_models.values():
        ensemble_pred += model_info['predictions']
    ensemble_pred /= len(top3_models)
    
    ensemble_corr, _ = pearsonr(y_val, ensemble_pred)
    print(f"简单平均集成验证相关系数: {ensemble_corr:.4f}")
    
    results['ensemble'] = {
        'val_corr': ensemble_corr,
        'predictions': ensemble_pred,
        'models': top3_models,
        'best_single': best_model
    }
    
    return results

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 极简稳定训练")
    print("🎯 回到最基础的方法，避免过拟合")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. 加载数据
        print("📊 加载数据...")
        train_path = "E:/CodeProjects/kaggleJoined/data/processed/drw-crypto-train_downsampled.parquet"
        train_df = pd.read_parquet(train_path)
        print(f"训练数据形状: {train_df.shape}")
        
        # 2. 最基础的预处理
        X = train_df.drop(columns=['label'])
        y = train_df['label'].fillna(train_df['label'].median())
        
        # 只做最基本的数据清理
        for col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())
        
        print("基础数据清理完成")
        
        # 3. 只选择原始特征 (不做任何特征工程)
        X_selected, selected_features = select_raw_features_only(X, y, n_features=15)
        
        # 4. 标准缩放
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        # 5. 时间感知分割
        split_point = int(len(X_scaled) * 0.8)
        X_train = X_scaled.iloc[:split_point]
        X_val = X_scaled.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_val = y.iloc[split_point:]
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
        
        # 6. 训练极简模型
        results = train_ultra_simple_models(X_train, y_train, X_val, y_val)
        
        # 7. 处理测试数据
        print("🔮 处理测试数据...")
        
        # 加载测试数据
        test_path = "E:/CodeProjects/kaggleJoined/data/raw/drw-crypto-market-prediction/test.parquet"
        test_df = pd.read_parquet(test_path)
        print(f"测试数据形状: {test_df.shape}")
        
        # 处理测试数据
        X_test = test_df.drop(columns=['label'])
        test_ids = test_df.index
        
        # 基础数据清理
        for col in selected_features:
            if col in X_test.columns:
                X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
                X_test[col] = X_test[col].fillna(X_test[col].median())
        
        # 选择相同特征
        X_test_selected = X_test[selected_features].copy()
        
        # 缩放
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_selected),
            columns=selected_features,
            index=X_test_selected.index
        )
        
        # 8. 预测
        if 'ensemble' in results and results['ensemble']['models']:
            top3_models = results['ensemble']['models']
            
            test_predictions = []
            
            for name, model_info in top3_models.items():
                try:
                    pred = model_info['model'].predict(X_test_scaled)
                    test_predictions.append(pred)
                    print(f"  {name}: 预测范围 [{pred.min():.4f}, {pred.max():.4f}]")
                except Exception as e:
                    print(f"  {name} 预测失败: {e}")
            
            if test_predictions:
                # 简单平均
                test_pred = np.mean(test_predictions, axis=0)
            else:
                print("❌ 所有模型预测失败，使用最佳单模型")
                test_pred = results['ensemble']['best_single']['model'].predict(X_test_scaled)
        else:
            print("❌ 集成失败，使用最佳单模型")
            test_pred = results['ensemble']['best_single']['model'].predict(X_test_scaled)
        
        print(f"\n测试集预测统计:")
        print(f"  均值: {test_pred.mean():.6f}")
        print(f"  标准差: {test_pred.std():.6f}")
        print(f"  范围: [{test_pred.min():.6f}, {test_pred.max():.6f}]")
        
        # 9. 创建提交文件
        submission_df = pd.DataFrame({
            'ID': test_ids,
            'prediction': test_pred
        })
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        submission_path = f"E:/CodeProjects/kaggleJoined/submissions/ultra_simple_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        
        # 10. 总结
        elapsed_time = time.time() - start_time
        ensemble_corr = results.get('ensemble', {}).get('val_corr', 0)
        best_single_corr = results.get('ensemble', {}).get('best_single', {}).get('val_corr', 0)
        
        print("\n" + "=" * 60)
        print("🎉 极简稳定训练完成!")
        print(f"⏱️  总耗时: {elapsed_time/60:.1f} 分钟")
        print(f"🎯 最佳单模型验证相关系数: {best_single_corr:.4f}")
        print(f"🎯 集成验证相关系数: {ensemble_corr:.4f}")
        print(f"📁 提交文件: {submission_path}")
        
        # 验证与样本文件的差异
        sample_path = "E:/CodeProjects/kaggleJoined/data/raw/drw-crypto-market-prediction/sample_submission.csv"
        sample_df = pd.read_csv(sample_path)
        correlation = np.corrcoef(test_pred, sample_df['prediction'])[0, 1]
        print(f"与样本提交的相关性: {correlation:.6f}")
        
        print("=" * 60)
        
        return {
            'ensemble_corr': ensemble_corr,
            'best_single_corr': best_single_corr,
            'submission_path': submission_path,
            'elapsed_time': elapsed_time,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
