"""
独立的模型序列化器

不依赖相对导入的独立版本，用于测试和演示。

作者：Augment Agent
"""

import os
import pickle
import json
import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd


class StandaloneModelSerializer:
    """
    独立模型序列化器
    
    负责模型的保存、加载和版本管理，不依赖项目架构。
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        """初始化模型序列化器"""
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.metadata_dir = self.output_dir / "metadata"
        
        # 创建输出目录
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, 
                  model: Any, 
                  model_name: str, 
                  model_type: str, 
                  metadata: Dict = None, 
                  version: str = None,
                  feature_names: List[str] = None) -> str:
        """
        保存模型及其元数据
        
        Args:
            model: 要保存的模型对象
            model_name: 模型名称
            model_type: 模型类型
            metadata: 模型元数据
            version: 模型版本
            feature_names: 特征名称列表
            
        Returns:
            保存的模型路径
        """
        # 生成版本号和时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if version is None:
            version = f"v{timestamp}"
        
        # 构建模型文件名
        model_filename = f"{model_name}_{model_type}_{version}"
        model_path = self.models_dir / f"{model_filename}.pkl"
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"模型已保存: {model_path}")
        
        # 准备元数据
        if metadata is None:
            metadata = {}
        
        # 添加基本元数据
        metadata.update({
            'model_name': model_name,
            'model_type': model_type,
            'version': version,
            'timestamp': timestamp,
            'format': 'pickle',
            'path': str(model_path),
            'feature_names': feature_names
        })
        
        # 添加模型指纹
        metadata['model_fingerprint'] = self._calculate_model_fingerprint(model_path)
        
        # 保存元数据
        metadata_path = self._save_metadata(metadata, model_filename)
        print(f"模型元数据已保存: {metadata_path}")
        
        return str(model_path)
    
    def load_model(self, 
                  model_path: str = None, 
                  model_name: str = None, 
                  model_type: str = None, 
                  version: str = None,
                  latest: bool = False) -> Tuple[Any, Dict]:
        """
        加载模型及其元数据
        
        Args:
            model_path: 模型文件路径
            model_name: 模型名称
            model_type: 模型类型
            version: 模型版本
            latest: 是否加载最新版本
            
        Returns:
            (model, metadata): 加载的模型对象和元数据
        """
        # 确定模型路径
        if model_path is None:
            if latest:
                model_path, metadata = self._find_latest_model(model_name, model_type)
                if model_path is None:
                    raise FileNotFoundError(f"未找到模型: {model_name}, {model_type}")
            else:
                if model_name is None or model_type is None:
                    raise ValueError("必须提供model_name和model_type，或者提供model_path")
                
                if version is None:
                    # 查找最新版本
                    model_path, metadata = self._find_latest_model(model_name, model_type)
                    if model_path is None:
                        raise FileNotFoundError(f"未找到模型: {model_name}, {model_type}")
                else:
                    # 使用指定版本
                    model_filename = f"{model_name}_{model_type}_{version}"
                    model_path = self.models_dir / f"{model_filename}.pkl"
                    metadata = self._load_metadata(model_filename)
        else:
            # 从路径中提取文件名
            model_filename = Path(model_path).stem
            metadata = self._load_metadata(model_filename)
        
        # 验证模型文件存在
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"模型已加载: {model_path}")
        return model, metadata
    
    def list_available_models(self, model_name: str = None, model_type: str = None) -> List[Dict]:
        """
        列出可用的模型
        
        Args:
            model_name: 过滤特定模型名称
            model_type: 过滤特定模型类型
            
        Returns:
            模型元数据列表
        """
        # 获取所有元数据文件
        metadata_files = list(self.metadata_dir.glob("*.json"))
        
        # 加载所有元数据
        all_metadata = []
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # 过滤条件
                if model_name is not None and metadata.get('model_name') != model_name:
                    continue
                
                if model_type is not None and metadata.get('model_type') != model_type:
                    continue
                
                # 验证模型文件存在
                model_path = metadata.get('path')
                if model_path and Path(model_path).exists():
                    metadata['exists'] = True
                else:
                    metadata['exists'] = False
                
                all_metadata.append(metadata)
                
            except Exception as e:
                print(f"加载元数据失败: {metadata_file}, 错误: {e}")
        
        # 按时间戳排序
        all_metadata.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return all_metadata
    
    def delete_model(self, model_path: str = None, model_name: str = None, 
                    model_type: str = None, version: str = None) -> bool:
        """
        删除模型及其元数据
        
        Args:
            model_path: 模型文件路径
            model_name: 模型名称
            model_type: 模型类型
            version: 模型版本
            
        Returns:
            是否成功删除
        """
        # 确定模型路径
        if model_path is None:
            if model_name is None or model_type is None or version is None:
                raise ValueError("必须提供model_path，或者提供model_name、model_type和version")
            
            model_filename = f"{model_name}_{model_type}_{version}"
            model_path = self.models_dir / f"{model_filename}.pkl"
        else:
            model_filename = Path(model_path).stem
        
        # 删除模型文件
        try:
            if Path(model_path).exists():
                Path(model_path).unlink()
                print(f"模型文件已删除: {model_path}")
            else:
                print(f"模型文件不存在: {model_path}")
                return False
            
            # 删除元数据文件
            metadata_path = self.metadata_dir / f"{model_filename}.json"
            
            if metadata_path.exists():
                metadata_path.unlink()
                print(f"元数据文件已删除: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"删除模型失败: {e}")
            return False
    
    def _save_metadata(self, metadata: Dict, model_filename: str) -> str:
        """保存模型元数据"""
        metadata_path = self.metadata_dir / f"{model_filename}.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(metadata_path)
    
    def _load_metadata(self, model_filename: str) -> Dict:
        """加载模型元数据"""
        metadata_path = self.metadata_dir / f"{model_filename}.json"
        
        if not metadata_path.exists():
            print(f"元数据文件不存在: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def _find_latest_model(self, model_name: str, model_type: str) -> Tuple[Optional[str], Dict]:
        """查找最新版本的模型"""
        all_metadata = self.list_available_models(model_name, model_type)
        
        if not all_metadata:
            return None, {}
        
        # 获取最新的元数据
        latest_metadata = all_metadata[0]
        
        # 检查模型文件是否存在
        model_path = latest_metadata.get('path')
        if model_path and Path(model_path).exists():
            return model_path, latest_metadata
        else:
            print(f"最新模型文件不存在: {model_path}")
            return None, {}
    
    def _calculate_model_fingerprint(self, model_path: str) -> str:
        """计算模型文件的指纹"""
        hasher = hashlib.md5()
        with open(model_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()


def test_standalone_serializer():
    """测试独立序列化器"""
    print("🧪 测试独立模型序列化器")
    print("=" * 50)
    
    # 创建序列化器
    serializer = StandaloneModelSerializer()
    print("✅ 序列化器创建成功")
    
    # 创建测试模型
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    # 生成测试数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    feature_names = [f'feature_{i}' for i in range(20)]
    
    # 训练模型
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    print("✅ 测试模型训练完成")
    
    # 准备元数据
    metadata = {
        'cv_scores': [0.85, 0.87, 0.86, 0.84, 0.88],
        'training_samples': len(X),
        'feature_count': len(feature_names),
        'performance': {
            'cv_auc': 0.86
        }
    }
    
    # 保存模型
    print("💾 保存测试模型...")
    model_path = serializer.save_model(
        model=model,
        model_name="test_logistic",
        model_type="sklearn",
        metadata=metadata,
        feature_names=feature_names
    )
    
    # 加载模型
    print("📂 加载测试模型...")
    loaded_model, loaded_metadata = serializer.load_model(
        model_name="test_logistic",
        model_type="sklearn",
        latest=True
    )
    
    # 验证模型
    print("🔍 验证模型...")
    original_pred = model.predict_proba(X[:5])
    loaded_pred = loaded_model.predict_proba(X[:5])
    
    if np.allclose(original_pred, loaded_pred):
        print("✅ 模型预测结果一致")
    else:
        print("❌ 模型预测结果不一致")
    
    # 列出可用模型
    print("📋 列出可用模型...")
    available_models = serializer.list_available_models()
    print(f"✅ 找到 {len(available_models)} 个模型")
    
    for model_info in available_models:
        print(f"  - {model_info['model_name']} ({model_info['model_type']})")
        print(f"    版本: {model_info['version']}")
        print(f"    存在: {'✅' if model_info['exists'] else '❌'}")
        if 'performance' in model_info:
            perf = model_info['performance']
            if 'cv_auc' in perf:
                print(f"    性能: AUC {perf['cv_auc']:.4f}")
    
    print("\n🎉 独立序列化器测试完成!")


if __name__ == "__main__":
    test_standalone_serializer()
