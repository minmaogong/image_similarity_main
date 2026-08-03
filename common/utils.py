import numpy as np
import torch
import os
import random
import re

# 实现一个统一设置随机数种子的函数，消除随机性
def seed_everything(seed):
    random.seed(seed) # 设置Python的随机种子，确保每次运行结果一致
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置Python的随机种子，确保每次运行结果一致
    np.random.seed(seed) # 设置numpy的随机种子，确保每次运行结果一致
    torch.manual_seed(seed) # 设置CPU的随机种子，确保每次运行结果一致
    torch.cuda.manual_seed(seed) # 设置CUDA的随机种子，确保每次运行结果一致
    torch.backends.cudnn.deterministic = True # 设置为True，确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # 关闭cudnn的自动调优功能，确保每次运行结果一致

def sorted_alphanum(img_names):
    convert = lambda str: int(str) if str.isdigit() else str.lower()
    alphanum_key = lambda name: [convert(x) for x in re.split(r'([0-9]+)', name)]
    return sorted(img_names, key=alphanum_key)

if __name__ == '__main__':
    imgs = os.listdir('./dataset')
    imgs = sorted_alphanum(imgs)
    print(imgs)
