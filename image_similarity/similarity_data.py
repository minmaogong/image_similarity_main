from torch.utils.data import Dataset, random_split
import torchvision.transforms as T
import os
from PIL import Image

from common.utils import sorted_alphanum
from similarity_config import *

class ImageDataset(Dataset):
    # 初始化
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(self.image_dir))

    # 获取数据集长度
    def __len__(self):
        return len(self.image_names)

    # 根据索引号获取元素
    def __getitem__(self, idx):
        # 1. 根据图片的完整访问路径
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        # 2. 打开图片
        image = Image.open(image_path)
        # 3. 应用转换操作，得到Tensor
        if self.transform:
            img_tensor = self.transform(image)
        else:
            raise ValueError("transform 参数不能为 None!")

        return img_tensor, img_tensor

# 创建数据集并划分
def create_dataset():
    # 定义图像转换操作（跳针大小，转为Tensor）
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
    ])
    # 创建数据集
    dataset = ImageDataset(image_dir=IMG_PATH, transform=transform)
    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = create_dataset()
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}")
