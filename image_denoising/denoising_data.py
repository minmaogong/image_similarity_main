import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from denoising_config import *
from common.utils import sorted_alphanum


class NoiseImageDataset(Dataset):
    # 初始化
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(self.image_dir))

    # 获取数据集长度
    def __len__(self):
        return len(self.image_names)

    # 根据索引号获取元素：(input, target) = (noise_image, image)
    def __getitem__(self, idx):
        # 1. 构建图片的完整访问路径
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        # 2. 打开图片
        image = Image.open(image_path).convert('RGB')
        # 3. 应用转换操作，得到Tensor
        if self.transform is not None:
            img_tensor = self.transform(image)
        else:
            raise ValueError("transform 参数不能为 None!")

        # 4. 加入噪声，得到模型真正输入
        img_noise_tensor = img_tensor + torch.randn_like(img_tensor) * NOISE_FACTOR
        # 将图片数据范围限制在(0, 1)
        img_noise_tensor = torch.clamp(img_noise_tensor, 0, 1) # clamp: 限制张量的值在指定范围内
        return img_noise_tensor, img_tensor # 返回噪声图片和原始图片 (input, target)

if __name__ == "__main__":
    dataset = NoiseImageDataset(IMG_PATH)
    print(dataset.image_names)
