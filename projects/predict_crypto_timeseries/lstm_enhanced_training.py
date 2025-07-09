"""
LSTM增强训练 - 基于现有lstm_model.py的改进版本
结合极简线性模型的成功经验和LSTM的时间序列能力
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import time
import os
import sys

# 导入现有LSTM模型
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from lstm_model import LSTMTrainer
    LSTM_AVAILABLE = True
    print("✅ 成功导入LSTM模型")
except ImportError as e:
    print(f"❌ LSTM导入失败: {e}")
    LSTM_AVAILABLE = False

def select_features_for_lstm(X, y, n_features=20):
    """为LSTM选择特征"""
    print(f"🎯 为LSTM选择{n_features}个特征...")
    
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
    
    # 选择最重要的特征
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    selected_features = [col for col, _ in sorted_features[:n_features]]
    
    print(f"最高相关性: {sorted_features[0][1]:.6f}")
    print(f"前5个特征: {selected_features[:5]}")
    
    return X[selected_features].copy(), selected_features

def train_lstm_ridge_ensemble(X_train, y_train, X_val, y_val):
    """训练LSTM+Ridge集成"""
    print("🤖 训练LSTM+Ridge集成...")
    
    results = {}
    
    # 1. Ridge基准模型
    print("训练Ridge基准...")
    ridge_model = Ridge(alpha=0.001, random_state=42)
    ridge_model.fit(X_train, y_train)
    
    ridge_pred = ridge_model.predict(X_val)
    ridge_corr, _ = pearsonr(y_val, ridge_pred)
    
    results['ridge'] = {
        'model': ridge_model, 
        'val_corr': ridge_corr, 
        'predictions': ridge_pred
    }
    print(f"  Ridge基准: {ridge_corr:.4f}")
    
    # 2. LSTM模型 (如果可用)
    if LSTM_AVAILABLE and len(X_train) > 100:
        print("训练LSTM模型...")
        try:
            # 使用保守的LSTM参数
            lstm_trainer = LSTMTrainer(
                window_size=10,        # 较短的时间窗口
                hidden_size=32,        # 较小的隐藏层
                num_layers=1,          # 单层LSTM
                dropout=0.3,           # 适度dropout
                bidirectional=False    # 单向LSTM
            )
            
            lstm_results = lstm_trainer.train(
                X_train, y_train, X_val, y_val,
                epochs=20,             # 较少的训练轮数
                batch_size=64,         # 适中的批次大小
                learning_rate=0.01,    # 较高的学习率
                patience=5             # 早停
            )
            
            if lstm_results and 'val_pearson' in lstm_results:
                # 预测验证集
                lstm_pred = lstm_trainer.predict(X_val)
                lstm_corr = lstm_results['val_pearson']
                
                results['lstm'] = {
                    'model': lstm_trainer, 
                    'val_corr': lstm_corr, 
                    'predictions': lstm_pred
                }
                print(f"  LSTM: {lstm_corr:.4f}")
            else:
                print("  LSTM训练失败")
                
        except Exception as e:
            print(f"  LSTM训练失败: {e}")
    
    # 3. 智能集成
    print("创建智能集成...")
    
    valid_models = {k: v for k, v in results.items() 
                   if 'val_corr' in v and not np.isnan(v['val_corr']) and v['val_corr'] > 0.05}
    
    if len(valid_models) == 0:
        print("❌ 没有有效的模型")
        return results
    
    if len(valid_models) == 1:
        # 只有一个模型
        model_name = list(valid_models.keys())[0]
        ensemble_pred = valid_models[model_name]['predictions']
        ensemble_corr = valid_models[model_name]['val_corr']
        print(f"只有一个有效模型: {model_name}")
    else:
        # 多个模型集成
        # 基于性能的加权平均
        total_weight = sum(max(0.01, model['val_corr']) for model in valid_models.values())
        ensemble_pred = np.zeros_like(y_val)
        
        weights = {}
        for name, model_info in valid_models.items():
            weight = max(0.01, model_info['val_corr']) / total_weight
            ensemble_pred += weight * model_info['predictions']
            weights[name] = weight
        
        ensemble_corr, _ = pearsonr(y_val, ensemble_pred)
        print(f"模型权重: {weights}")
    
    print(f"集成验证相关系数: {ensemble_corr:.4f}")
    
    results['ensemble'] = {
        'val_corr': ensemble_corr,
        'predictions': ensemble_pred,
        'models': valid_models
    }
    
    return results

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 LSTM增强训练")
    print("🎯 结合线性模型稳定性和LSTM时间序列能力")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. 加载数据
        print("📊 加载数据...")
        train_path = "E:/CodeProjects/kaggleJoined/data/processed/drw-crypto-train_downsampled.parquet"
        train_df = pd.read_parquet(train_path)
        print(f"训练数据形状: {train_df.shape}")
        
        # 2. 基础预处理
        X = train_df.drop(columns=['label'])
        y = train_df['label'].fillna(train_df['label'].median())
        
        # 数据清理
        for col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())
        
        print("数据清理完成")
        
        # 3. 特征选择
        X_selected, selected_features = select_features_for_lstm(X, y, n_features=20)
        
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
        
        # 6. 训练LSTM+Ridge集成
        results = train_lstm_ridge_ensemble(X_train, y_train, X_val, y_val)
        
        # 7. 处理测试数据
        print("🔮 处理测试数据...")
        
        # 加载测试数据
        test_path = "E:/CodeProjects/kaggleJoined/data/raw/drw-crypto-market-prediction/test.parquet"
        test_df = pd.read_parquet(test_path)
        print(f"测试数据形状: {test_df.shape}")
        
        # 处理测试数据
        X_test = test_df.drop(columns=['label'])
        test_ids = test_df.index
        
        # 数据清理
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
            ensemble_models = results['ensemble']['models']
            
            test_predictions = []
            weights = []
            
            for name, model_info in ensemble_models.items():
                try:
                    if name == 'lstm':
                        # LSTM预测
                        pred = model_info['model'].predict(X_test_scaled)
                    else:
                        # Ridge预测
                        pred = model_info['model'].predict(X_test_scaled)
                    
                    test_predictions.append(pred)
                    weights.append(max(0.01, model_info['val_corr']))
                    print(f"  {name}: 预测范围 [{pred.min():.4f}, {pred.max():.4f}]")
                except Exception as e:
                    print(f"  {name} 预测失败: {e}")
            
            if test_predictions:
                # 加权平均
                total_weight = sum(weights)
                test_pred = np.zeros(len(X_test_scaled))
                
                for pred, weight in zip(test_predictions, weights):
                    test_pred += (weight / total_weight) * pred
            else:
                print("❌ 所有模型预测失败，使用Ridge基准")
                test_pred = results['ridge']['model'].predict(X_test_scaled)
        else:
            print("❌ 集成失败，使用Ridge基准")
            test_pred = results['ridge']['model'].predict(X_test_scaled)
        
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
        submission_path = f"E:/CodeProjects/kaggleJoined/submissions/lstm_enhanced_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        
        # 10. 总结
        elapsed_time = time.time() - start_time
        ensemble_corr = results.get('ensemble', {}).get('val_corr', 0)
        
        print("\n" + "=" * 60)
        print("🎉 LSTM增强训练完成!")
        print(f"⏱️  总耗时: {elapsed_time/60:.1f} 分钟")
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
