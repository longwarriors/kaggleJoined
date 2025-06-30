import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import save_checkpoint, load_checkpoint
from datasets.cats_dogs import labelled_dogcat_set, inference_dogcat_set

print(len(labelled_dogcat_set))
print(len(inference_dogcat_set))
