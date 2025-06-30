import sys
import os


# 添加项目根目录到Python路径的更健壮方法
def get_project_root():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)

    # 向上查找，直到找到包含 'datasets' 和 'scripts' 目录的根目录
    while current_dir != os.path.dirname(current_dir):  # 避免到达文件系统根目录
        if os.path.exists(os.path.join(current_dir, "datasets")) and os.path.exists(
            os.path.join(current_dir, "scripts")
        ):
            return current_dir
        current_dir = os.path.dirname(current_dir)

    # 如果没找到，使用固定的向上三级方法作为备选
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


project_root = get_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts import save_checkpoint, load_checkpoint
from datasets.cats_dogs import labelled_dogcat_set, inference_dogcat_set

print(len(labelled_dogcat_set))
print(len(inference_dogcat_set))
